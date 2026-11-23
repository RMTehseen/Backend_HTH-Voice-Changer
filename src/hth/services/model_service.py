"""Model lifecycle service — manages imports, quarantine, registration, consent, and deletion."""

from __future__ import annotations

import datetime
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hth.core.logging import get_logger
from hth.core.settings import get_settings
from hth.domain.model_contracts import (
    AssetDescriptor,
    CompatibilityStatus,
    ConsentAttestationRequest,
    ConsentRecord,
    ModelAuditEntry,
    ModelAuditEventType,
    ModelFormat,
    ModelManifestMetadata,
    ModelRecord,
    ModelSummary,
    SourceType,
)
from hth.repositories.models import AuditRow, ConsentRow, ModelMetaRow
from hth.services.archive_validator import ArchiveSecurityValidator
from hth.services.license_service import LicenseService
from hth.services.model_parsers import get_parser_for_extracted_directory

logger = get_logger("hth.services.model")


class ModelNotFoundError(Exception):
    """Raised when a requested model_id does not exist."""


class ModelLifecycleError(Exception):
    """Raised when a model lifecycle action violates state or disk constraints."""


class ModelService:
    """Orchestrates model validation, quarantine extraction, database registration, consent, and deletion."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        storage_dir: str | Path | None = None,
    ) -> None:
        self._session_factory = session_factory
        if storage_dir is None:
            settings = get_settings()
            db_dir = os.path.dirname(settings.db_path)
            self._storage_dir = Path(db_dir) / "models"
        else:
            self._storage_dir = Path(storage_dir)

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._quarantine_root = self._storage_dir.parent / "quarantine"
        self._quarantine_root.mkdir(parents=True, exist_ok=True)

    async def record_audit_event(
        self,
        event_type: ModelAuditEventType | str,
        model_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        """Record an immutable audit entry in the database."""
        event_str = (
            event_type.value if isinstance(event_type, ModelAuditEventType) else str(event_type)
        )
        async with self._session_factory() as session:
            row = AuditRow(
                event_type=event_str,
                model_id=model_id,
                payload=json.dumps(payload),
                occurred_at=datetime.datetime.utcnow(),
            )
            session.add(row)
            await session.commit()

    async def import_archive(
        self,
        archive_path: str | Path,
        original_filename: str,
        source_type: SourceType = SourceType.IMPORTED_ARCHIVE,
    ) -> ModelRecord:
        """Validate, quarantine-extract, parse, and persist a model archive."""
        archive_path = Path(archive_path)
        quarantine_id = f"quarantine_{uuid.uuid4().hex[:12]}"
        quarantine_dir = self._quarantine_root / quarantine_id

        await self.record_audit_event(
            ModelAuditEventType.IMPORT_ATTEMPTED,
            model_id=None,
            payload={"filename": original_filename, "archive_size": archive_path.stat().st_size},
        )

        try:
            # 1. Archive extraction & security checks
            suffix = archive_path.suffix.lower()
            if suffix == ".zip" or original_filename.lower().endswith(".zip"):
                ArchiveSecurityValidator.validate_and_extract_zip(archive_path, quarantine_dir)
            elif suffix in (".tar", ".gz", ".tgz") or any(
                original_filename.lower().endswith(ext) for ext in (".tar", ".tar.gz", ".tgz")
            ):
                ArchiveSecurityValidator.validate_and_extract_tar(archive_path, quarantine_dir)
            else:
                raise ModelLifecycleError(
                    f"Unsupported archive extension: {suffix}. Supported: .zip, .tar, .tar.gz"
                )

            # 2. Match format parser
            parser = get_parser_for_extracted_directory(str(quarantine_dir))
            if parser is None:
                raise ModelLifecycleError(
                    "Unrecognized or unsupported model format. Directory does not contain valid "
                    "Beatrice paraphernalia, RVC/Applio weights, or synthetic model descriptors."
                )

            # 3. Parse metadata
            parsed = parser.parse(str(quarantine_dir))

            # 4. Celebrity / Public figure likeness heuristic check
            is_celebrity = LicenseService.check_public_figure_likeness(
                f"{parsed.name} {parsed.description}"
            )

            # 5. Determine initial compatibility status
            if is_celebrity:
                compat_status = CompatibilityStatus.BLOCKED_COMMERCIAL
            elif not parsed.commercial_eligible and LicenseService.is_commercial_release_mode():
                compat_status = CompatibilityStatus.BLOCKED_COMMERCIAL
            else:
                compat_status = CompatibilityStatus.CONSENT_REQUIRED

            # 6. Generate final model_id and move from quarantine
            model_id = f"{parsed.engine_id}_{uuid.uuid4().hex[:8]}"
            final_install_path = self._storage_dir / model_id

            if final_install_path.exists():
                shutil.rmtree(final_install_path, ignore_errors=True)
            shutil.move(str(quarantine_dir), str(final_install_path))

            # 7. Rebase asset paths to final install path
            total_size = sum(a.size_bytes for a in parsed.assets)
            primary_hash = parsed.assets[0].sha256 if parsed.assets else ""

            model_record = ModelRecord(
                model_id=model_id,
                engine_id=parsed.engine_id,
                model_format=parsed.model_format,
                name=parsed.name,
                version=parsed.version,
                description=parsed.description,
                manifest=parsed.manifest,
                asset_list=parsed.assets,
                primary_hash=primary_hash,
                size_bytes=total_size,
                sample_rate=parsed.sample_rate,
                license_identifier=parsed.license_identifier,
                license_source_url=parsed.license_source_url,
                source_type=source_type,
                owner_consent_evidence="",
                commercial_eligible=parsed.commercial_eligible,
                is_public_figure_detected=is_celebrity,
                compatibility_status=compat_status,
                install_path=str(final_install_path),
                import_date=datetime.datetime.utcnow(),
                is_active=False,
            )

            # 8. Save to DB
            async with self._session_factory() as session:
                row = ModelMetaRow(
                    model_id=model_record.model_id,
                    engine_id=model_record.engine_id,
                    model_format=model_record.model_format.value,
                    name=model_record.name,
                    version=model_record.version,
                    description=model_record.description,
                    manifest_json=model_record.manifest.model_dump_json(),
                    asset_list_json=json.dumps([a.model_dump() for a in model_record.asset_list]),
                    sha256=model_record.primary_hash,
                    size_bytes=model_record.size_bytes,
                    sample_rate=model_record.sample_rate,
                    license_identifier=model_record.license_identifier,
                    license_source_url=model_record.license_source_url,
                    source_type=model_record.source_type.value,
                    owner_consent_evidence=model_record.owner_consent_evidence,
                    commercial_eligible=model_record.commercial_eligible,
                    is_public_figure_detected=model_record.is_public_figure_detected,
                    compatibility_status=model_record.compatibility_status.value,
                    install_path=model_record.install_path,
                    is_active=False,
                )
                session.add(row)
                await session.commit()

            await self.record_audit_event(
                ModelAuditEventType.IMPORT_SUCCESS,
                model_id=model_id,
                payload={
                    "name": model_record.name,
                    "engine_id": model_record.engine_id,
                    "format": model_record.model_format.value,
                    "license": model_record.license_identifier,
                    "commercial_eligible": model_record.commercial_eligible,
                },
            )

            return model_record

        except Exception as exc:
            shutil.rmtree(quarantine_dir, ignore_errors=True)
            await self.record_audit_event(
                ModelAuditEventType.IMPORT_REJECTED,
                model_id=None,
                payload={"filename": original_filename, "error": str(exc)},
            )
            raise

    async def record_consent(
        self,
        model_id: str,
        request: ConsentAttestationRequest,
    ) -> ConsentRecord:
        """Validate and record legal consent attestation for a model."""
        model = await self.get_model(model_id)

        # Validate legal attestation checklist
        LicenseService.validate_attestation(request, model_name=model.name)

        consent_record = ConsentRecord(
            model_id=model_id,
            attested_by=request.attested_by,
            attested_at=datetime.datetime.utcnow(),
            is_own_voice=request.is_own_voice,
            has_written_consent=request.has_written_consent,
            has_commercial_rights=request.has_commercial_rights,
            no_celebrity_likeness=request.no_celebrity_likeness,
            notes=request.notes,
        )

        async with self._session_factory() as session:
            # 1. Insert consent row
            crow = ConsentRow(
                model_id=consent_record.model_id,
                attested_by=consent_record.attested_by,
                attested_at=consent_record.attested_at,
                is_own_voice=consent_record.is_own_voice,
                has_written_consent=consent_record.has_written_consent,
                has_commercial_rights=consent_record.has_commercial_rights,
                no_celebrity_likeness=consent_record.no_celebrity_likeness,
                license_accepted=request.license_accepted,
                notes=consent_record.notes,
            )
            session.add(crow)

            # 2. Update model compatibility status to COMPATIBLE
            new_status = CompatibilityStatus.COMPATIBLE
            if not model.commercial_eligible and LicenseService.is_commercial_release_mode():
                new_status = CompatibilityStatus.BLOCKED_COMMERCIAL

            await session.execute(
                update(ModelMetaRow)
                .where(ModelMetaRow.model_id == model_id)
                .values(
                    compatibility_status=new_status.value,
                    owner_consent_evidence=f"Attested by {request.attested_by} on {consent_record.attested_at.isoformat()}",
                )
            )
            await session.commit()

        await self.record_audit_event(
            ModelAuditEventType.CONSENT_ATTESTED,
            model_id=model_id,
            payload={
                "attested_by": request.attested_by,
                "is_own_voice": request.is_own_voice,
                "has_written_consent": request.has_written_consent,
                "no_celebrity_likeness": request.no_celebrity_likeness,
            },
        )

        return consent_record

    async def get_model(self, model_id: str) -> ModelRecord:
        """Fetch full model record by model_id."""
        async with self._session_factory() as session:
            stmt = select(ModelMetaRow).where(ModelMetaRow.model_id == model_id)
            res = await session.execute(stmt)
            row = res.scalar_one_or_none()
            if row is None:
                raise ModelNotFoundError(f"Model with id '{model_id}' was not found")

            manifest_data = json.loads(row.manifest_json) if row.manifest_json else {}
            assets_data = json.loads(row.asset_list_json) if row.asset_list_json else []

            return ModelRecord(
                model_id=row.model_id,
                engine_id=row.engine_id,
                model_format=ModelFormat(row.model_format),
                name=row.name,
                version=row.version,
                description=row.description,
                manifest=ModelManifestMetadata(**manifest_data),
                asset_list=[AssetDescriptor(**a) for a in assets_data],
                primary_hash=row.sha256,
                size_bytes=row.size_bytes,
                sample_rate=row.sample_rate,
                license_identifier=row.license_identifier,
                license_source_url=row.license_source_url,
                source_type=SourceType(row.source_type),
                owner_consent_evidence=row.owner_consent_evidence,
                commercial_eligible=row.commercial_eligible,
                is_public_figure_detected=row.is_public_figure_detected,
                compatibility_status=CompatibilityStatus(row.compatibility_status),
                install_path=row.install_path,
                import_date=row.created_at,
                is_active=row.is_active,
            )

    async def list_models(self) -> list[ModelSummary]:
        """List registered voice models with consent and active status."""
        async with self._session_factory() as session:
            models_res = await session.execute(
                select(ModelMetaRow).order_by(ModelMetaRow.created_at.desc())
            )
            model_rows = models_res.scalars().all()

            # Query consents
            consents_res = await session.execute(select(ConsentRow.model_id).distinct())
            consented_ids = set(consents_res.scalars().all())

            summaries: list[ModelSummary] = []
            for r in model_rows:
                summaries.append(
                    ModelSummary(
                        model_id=r.model_id,
                        engine_id=r.engine_id,
                        model_format=ModelFormat(r.model_format),
                        name=r.name,
                        version=r.version,
                        description=r.description,
                        size_bytes=r.size_bytes,
                        sample_rate=r.sample_rate,
                        license_identifier=r.license_identifier,
                        commercial_eligible=r.commercial_eligible,
                        compatibility_status=CompatibilityStatus(r.compatibility_status),
                        has_consent=r.model_id in consented_ids,
                        is_active=r.is_active,
                        import_date=r.created_at,
                    )
                )
            return summaries

    async def delete_model(self, model_id: str) -> None:
        """Unregister model, delete application-managed disk assets, and log audit event."""
        model = await self.get_model(model_id)

        if model.is_active:
            raise ModelLifecycleError(
                f"Cannot delete model '{model.name}' while it is currently active. Switch models first."
            )

        # 1. Delete app-managed files on disk
        install_path = Path(model.install_path)
        if install_path.exists() and install_path.is_dir():
            # Confirm path is safely inside our managed storage directory
            try:
                install_path.resolve().relative_to(self._storage_dir.resolve())
                shutil.rmtree(install_path, ignore_errors=True)
            except ValueError:
                logger.warning(
                    "Model install path is outside app models storage dir, skipping rmtree",
                    path=str(install_path),
                )

        # 2. Delete database rows
        async with self._session_factory() as session:
            await session.execute(delete(ConsentRow).where(ConsentRow.model_id == model_id))
            await session.execute(delete(ModelMetaRow).where(ModelMetaRow.model_id == model_id))
            await session.commit()

        # 3. Record audit event
        await self.record_audit_event(
            ModelAuditEventType.MODEL_DELETED,
            model_id=model_id,
            payload={
                "name": model.name,
                "size_bytes": model.size_bytes,
                "install_path": model.install_path,
            },
        )

    async def export_model(self, model_id: str, target_zip: Path) -> Path:
        """Export model package to an external ZIP archive."""
        model = await self.get_model(model_id)
        install_path = Path(model.install_path)
        if not install_path.exists():
            raise ModelLifecycleError(f"Model assets missing on disk at {install_path}")

        target_zip.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(target_zip.with_suffix("")), "zip", str(install_path))
        return target_zip

    async def get_audit_trail(self, model_id: str | None = None) -> list[ModelAuditEntry]:
        """Fetch audit events from the audit log."""
        async with self._session_factory() as session:
            stmt = select(AuditRow).order_by(AuditRow.occurred_at.desc())
            if model_id is not None:
                stmt = stmt.where(AuditRow.model_id == model_id)
            res = await session.execute(stmt)
            rows = res.scalars().all()
            entries: list[ModelAuditEntry] = []
            for r in rows:
                try:
                    p = json.loads(r.payload)
                except Exception:
                    p = {"raw": r.payload}
                entries.append(
                    ModelAuditEntry(
                        id=r.id,
                        event_type=r.event_type,
                        model_id=r.model_id,
                        payload=p,
                        occurred_at=r.occurred_at,
                    )
                )
            return entries
