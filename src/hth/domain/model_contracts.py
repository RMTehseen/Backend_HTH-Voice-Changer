"""Provider-neutral model registry contracts, schemas, and enums."""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelFormat(str, Enum):
    BEATRICE_PARAPHERNALIA = "beatrice_paraphernalia"
    RVC_PTH_INDEX = "rvc_pth_index"
    ONNX = "onnx"
    SYNTHETIC_TEST = "synthetic_test"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    SELF_RECORDED = "self_recorded"
    CUSTOM_LICENSED = "custom_licensed"
    VENDOR_SUPPLIED = "vendor_supplied"
    IMPORTED_ARCHIVE = "imported_archive"


class CompatibilityStatus(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE_ENGINE = "incompatible_engine"
    INCOMPATIBLE_HARDWARE = "incompatible_hardware"
    BLOCKED_COMMERCIAL = "blocked_commercial"
    CONSENT_REQUIRED = "consent_required"
    CORRUPT = "corrupt"


class ModelAuditEventType(str, Enum):
    IMPORT_ATTEMPTED = "MODEL_IMPORT_ATTEMPTED"
    IMPORT_SUCCESS = "MODEL_IMPORT_SUCCESS"
    IMPORT_REJECTED = "MODEL_IMPORT_REJECTED"
    CONSENT_ATTESTED = "MODEL_CONSENT_ATTESTED"
    MODEL_ACTIVATED = "MODEL_ACTIVATED"
    MODEL_SWITCH_FAILED = "MODEL_SWITCH_FAILED"
    MODEL_DELETED = "MODEL_DELETED"


class AssetDescriptor(BaseModel):
    relative_path: str
    sha256: str
    size_bytes: int
    mime_type: str = "application/octet-stream"
    is_required: bool = True


class ModelManifestMetadata(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    speaker_count: int = 1
    speakers: list[dict[str, Any]] = Field(default_factory=list)
    target_sample_rate: int = 16000
    extra_params: dict[str, Any] = Field(default_factory=dict)


class ConsentAttestationRequest(BaseModel):
    attested_by: str = Field(
        ..., min_length=1, description="Legal name or username of person attesting"
    )
    is_own_voice: bool = Field(False, description="User certifies this voice likeness is their own")
    has_written_consent: bool = Field(
        False, description="User has explicit written consent from the voice donor"
    )
    has_commercial_rights: bool = Field(False, description="Rights allow commercial exploitation")
    no_celebrity_likeness: bool = Field(
        False, description="Certifies NOT a celebrity/public figure/copyrighted character"
    )
    license_accepted: bool = Field(
        False, description="Explicit acceptance of underlying model terms"
    )
    notes: str = ""


class ConsentRecord(BaseModel):
    model_id: str
    attested_by: str
    attested_at: datetime.datetime
    is_own_voice: bool
    has_written_consent: bool
    has_commercial_rights: bool
    no_celebrity_likeness: bool
    notes: str = ""


class ModelRecord(BaseModel):
    model_id: str
    engine_id: str
    model_format: ModelFormat
    name: str
    version: str = "1.0.0"
    description: str = ""
    manifest: ModelManifestMetadata
    asset_list: list[AssetDescriptor] = Field(default_factory=list)
    primary_hash: str
    size_bytes: int
    sample_rate: int = 16000
    license_identifier: str
    license_source_url: str = ""
    source_type: SourceType = SourceType.IMPORTED_ARCHIVE
    owner_consent_evidence: str = ""
    commercial_eligible: bool = False
    is_public_figure_detected: bool = False
    compatibility_status: CompatibilityStatus = CompatibilityStatus.CONSENT_REQUIRED
    install_path: str
    import_date: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    is_active: bool = False


class ModelSummary(BaseModel):
    model_id: str
    engine_id: str
    model_format: ModelFormat
    name: str
    version: str
    description: str
    size_bytes: int
    sample_rate: int
    license_identifier: str
    commercial_eligible: bool
    compatibility_status: CompatibilityStatus
    has_consent: bool
    is_active: bool
    import_date: datetime.datetime


class ModelSwitchResult(BaseModel):
    previous_model_id: str | None
    active_model_id: str
    engine_state: str
    success: bool
    message: str


class ModelAuditEntry(BaseModel):
    id: int
    event_type: str
    model_id: str | None
    payload: dict[str, Any]
    occurred_at: datetime.datetime
