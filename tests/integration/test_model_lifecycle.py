"""Integration tests for Gate 7: full model lifecycle, security, consent, deletion, and audit trail."""

import json
import zipfile
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hth.adapters.fake_engine import FakeEngine
from hth.core.database import Base
from hth.domain.model_contracts import (
    CompatibilityStatus,
    ConsentAttestationRequest,
    ModelAuditEventType,
)
from hth.services.archive_validator import ArchiveSecurityError
from hth.services.engine_service import EngineService
from hth.services.license_service import LicensingPolicyViolation
from hth.services.model_service import ModelService
from hth.services.switching_service import SwitchingService


@pytest.fixture
async def async_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield session_factory
    await engine.dispose()


@pytest.fixture
def test_storage(tmp_path):
    storage_dir = tmp_path / "models"
    storage_dir.mkdir(parents=True)
    return storage_dir


def create_sample_synthetic_archive(
    zip_path: Path, name: str = "Test Voice", commercial: bool = True
) -> Path:
    with zipfile.ZipFile(zip_path, "w") as zf:
        manifest = {
            "name": name,
            "version": "1.0.0",
            "description": "Integration test model",
            "sample_rate": 48000,
            "license": "MIT" if commercial else "Academic-NonCommercial",
            "commercial_eligible": commercial,
            "speaker_count": 1,
        }
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("weights.bin", b"\x00\x01\x02\x03" * 128)
    return zip_path


@pytest.mark.asyncio
async def test_full_model_lifecycle_gate7(async_db, test_storage, tmp_path):
    """Gate 7 Verification: Import, security validation, consent, switching, deletion, and audit trail."""
    model_service = ModelService(async_db, storage_dir=test_storage)
    fake_engine = FakeEngine()
    engine_service = EngineService(fake_engine)
    switching_service = SwitchingService(model_service, engine_service, async_db)

    # 1. Reject malicious archive with Zip Slip
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../traversal.txt", b"evil")

    with pytest.raises(ArchiveSecurityError):
        await model_service.import_archive(evil_zip, "evil.zip")

    # Verify no leaked files in storage or quarantine
    assert len(list(test_storage.iterdir())) == 0

    # 2. Import valid model archive
    good_zip = tmp_path / "model1.zip"
    create_sample_synthetic_archive(good_zip, name="Sample Voice", commercial=True)

    model = await model_service.import_archive(good_zip, "model1.zip")
    assert model.name == "Sample Voice"
    assert model.compatibility_status == CompatibilityStatus.CONSENT_REQUIRED
    assert Path(model.install_path).exists()
    assert (Path(model.install_path) / "manifest.json").exists()

    # 3. Model switching blocked before consent attestation
    with pytest.raises(LicensingPolicyViolation, match="requires signed consent attestation"):
        await switching_service.switch_model(model.model_id)

    # 4. Attest legal consent
    attestation = ConsentAttestationRequest(
        attested_by="Alice Developer",
        is_own_voice=True,
        has_written_consent=False,
        has_commercial_rights=True,
        no_celebrity_likeness=True,
        license_accepted=True,
        notes="Self-recorded dataset",
    )
    consent = await model_service.record_consent(model.model_id, attestation)
    assert consent.attested_by == "Alice Developer"
    assert consent.is_own_voice is True

    # Check model updated to COMPATIBLE
    updated_model = await model_service.get_model(model.model_id)
    assert updated_model.compatibility_status == CompatibilityStatus.COMPATIBLE

    # 5. Explicit model switch
    switch_res = await switching_service.switch_model(model.model_id)
    assert switch_res.success is True
    assert switch_res.active_model_id == model.model_id

    # Verify marked active in DB
    refreshed_list = await model_service.list_models()
    assert any(m.model_id == model.model_id and m.is_active for m in refreshed_list)

    # 6. Cannot delete active model
    with pytest.raises(Exception, match="Cannot delete model .* while it is currently active"):
        await model_service.delete_model(model.model_id)

    # Switch away to allow deletion
    model2_zip = tmp_path / "model2.zip"
    create_sample_synthetic_archive(model2_zip, name="Second Voice", commercial=True)
    model2 = await model_service.import_archive(model2_zip, "model2.zip")
    await model_service.record_consent(model2.model_id, attestation)
    await switching_service.switch_model(model2.model_id)

    # 7. Delete first model — verify app-managed files wiped from disk
    model1_install_path = Path(model.install_path)
    assert model1_install_path.exists()
    await model_service.delete_model(model.model_id)

    # Files must be gone from disk
    assert not model1_install_path.exists()

    # DB record must be gone
    models_remaining = await model_service.list_models()
    assert not any(m.model_id == model.model_id for m in models_remaining)

    # 8. Verify audit trail has recorded every event
    audit_events = await model_service.get_audit_trail()
    event_types = [e.event_type for e in audit_events]

    assert ModelAuditEventType.IMPORT_REJECTED.value in event_types
    assert ModelAuditEventType.IMPORT_ATTEMPTED.value in event_types
    assert ModelAuditEventType.IMPORT_SUCCESS.value in event_types
    assert ModelAuditEventType.CONSENT_ATTESTED.value in event_types
    assert ModelAuditEventType.MODEL_ACTIVATED.value in event_types
    assert ModelAuditEventType.MODEL_DELETED.value in event_types
