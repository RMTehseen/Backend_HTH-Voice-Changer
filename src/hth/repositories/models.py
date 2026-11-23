"""SQLAlchemy ORM models for HTH persistence layer."""

import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from hth.core.database import Base

__all__ = ["SettingRow", "ModelMetaRow", "ConsentRow", "AuditRow"]


class SettingRow(Base):
    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=True,
    )


class ModelMetaRow(Base):
    __tablename__ = "model_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    engine_id: Mapped[str] = mapped_column(String(64), nullable=False)
    model_format: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(32), default="1.0.0", nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    manifest_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    asset_list_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, default=16000, nullable=False)
    license_identifier: Mapped[str] = mapped_column(String(64), nullable=False)
    license_source_url: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), default="imported_archive", nullable=False)
    owner_consent_evidence: Mapped[str] = mapped_column(Text, default="", nullable=False)
    commercial_eligible: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_public_figure_detected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    compatibility_status: Mapped[str] = mapped_column(
        String(64), default="consent_required", nullable=False
    )
    install_path: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )


class ConsentRow(Base):
    __tablename__ = "consent"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    attested_by: Mapped[str] = mapped_column(String(255), nullable=False)
    attested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    is_own_voice: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_written_consent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_commercial_rights: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    no_celebrity_likeness: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    license_accepted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=True)


class AuditRow(Base):
    __tablename__ = "audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    model_id: Mapped[str] = mapped_column(String(255), nullable=True)
    payload: Mapped[str] = mapped_column(Text, nullable=False)  # JSON text
    occurred_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
