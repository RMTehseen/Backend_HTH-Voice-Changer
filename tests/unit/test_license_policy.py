"""Unit tests for licensing and consent policy engine."""

import pytest

from hth.domain.model_contracts import (
    CompatibilityStatus,
    ConsentAttestationRequest,
    ModelFormat,
    ModelManifestMetadata,
    ModelRecord,
)
from hth.services.license_service import LicenseService, LicensingPolicyViolation


def test_attestation_requires_name():
    req = ConsentAttestationRequest(
        attested_by="   ",
        is_own_voice=True,
        no_celebrity_likeness=True,
        license_accepted=True,
    )
    with pytest.raises(LicensingPolicyViolation, match="valid name"):
        LicenseService.validate_attestation(req)


def test_attestation_requires_license_acceptance():
    req = ConsentAttestationRequest(
        attested_by="Jane Doe",
        is_own_voice=True,
        no_celebrity_likeness=True,
        license_accepted=False,
    )
    with pytest.raises(LicensingPolicyViolation, match="accept the underlying model license"):
        LicenseService.validate_attestation(req)


def test_attestation_requires_rights_or_consent():
    req = ConsentAttestationRequest(
        attested_by="Jane Doe",
        is_own_voice=False,
        has_written_consent=False,
        has_commercial_rights=False,
        no_celebrity_likeness=True,
        license_accepted=True,
    )
    with pytest.raises(LicensingPolicyViolation, match="own, backed by explicit written consent"):
        LicenseService.validate_attestation(req)


def test_celebrity_detection_blocks_attestation():
    req = ConsentAttestationRequest(
        attested_by="Jane Doe",
        is_own_voice=True,
        no_celebrity_likeness=True,
        license_accepted=True,
    )
    with pytest.raises(
        LicensingPolicyViolation, match="Public figure or celebrity likeness detected"
    ):
        LicenseService.validate_attestation(req, model_name="Joe Biden Voice Model")


def test_jvs_blocked_in_commercial_mode(monkeypatch):
    """JVS models must be hard-blocked from activation in commercial mode."""
    monkeypatch.setenv("HTH_COMMERCIAL_RELEASE", "1")

    jvs_model = ModelRecord(
        model_id="beatrice_jvs_001",
        engine_id="beatrice",
        model_format=ModelFormat.BEATRICE_PARAPHERNALIA,
        name="JVS Corpus Voice",
        manifest=ModelManifestMetadata(name="JVS Corpus Voice"),
        primary_hash="abc",
        size_bytes=1000,
        license_identifier="JVS-NonCommercial",
        commercial_eligible=False,
        install_path="/tmp/jvs",
    )

    with pytest.raises(
        LicensingPolicyViolation,
        match="Commercial mode strictly forbids non-commercial academic models",
    ):
        LicenseService.check_activation_eligibility(jvs_model, commercial_mode=True)


def test_beatrice_blocked_in_commercial_mode_without_auth(monkeypatch):
    """Beatrice engine models must be blocked in commercial release unless authorization is recorded."""
    monkeypatch.setenv("HTH_COMMERCIAL_RELEASE", "1")
    monkeypatch.delenv("HTH_BEATRICE_COMMERCIAL_AUTH", raising=False)

    beatrice_custom = ModelRecord(
        model_id="beatrice_custom_001",
        engine_id="beatrice",
        model_format=ModelFormat.BEATRICE_PARAPHERNALIA,
        name="Custom Beatrice Voice",
        manifest=ModelManifestMetadata(name="Custom Beatrice Voice"),
        primary_hash="def",
        size_bytes=1000,
        license_identifier="Commercial-Custom",
        commercial_eligible=True,
        install_path="/tmp/beatrice",
    )

    with pytest.raises(
        LicensingPolicyViolation,
        match="Beatrice DSP engine contains proprietary VST inference libraries",
    ):
        LicenseService.check_activation_eligibility(beatrice_custom, commercial_mode=True)


def test_commercial_eligible_model_allowed(monkeypatch):
    """Permissive commercial model (e.g. Synthetic / FakeEngine) allowed in commercial mode."""
    commercial_model = ModelRecord(
        model_id="fake_001",
        engine_id="fake",
        model_format=ModelFormat.SYNTHETIC_TEST,
        name="Permissive Voice",
        manifest=ModelManifestMetadata(name="Permissive Voice"),
        primary_hash="ghi",
        size_bytes=500,
        license_identifier="MIT",
        commercial_eligible=True,
        compatibility_status=CompatibilityStatus.COMPATIBLE,
        install_path="/tmp/fake",
    )

    # Should not raise
    LicenseService.check_activation_eligibility(commercial_model, commercial_mode=True)
