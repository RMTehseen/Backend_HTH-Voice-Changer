"""License and consent policy engine — enforces commercial, consent, and likeness rules."""

from __future__ import annotations

import os
import re

from hth.domain.model_contracts import (
    ConsentAttestationRequest,
    ModelRecord,
)

# Heuristic list of public figures, politicians, and copyright-restricted likenesses
PUBLIC_FIGURE_PATTERNS = [
    r"\bbiden\b",
    r"\btrump\b",
    r"\bobama\b",
    r"\bmusk\b",
    r"\btaylor\s+swift\b",
    r"\bdrake\b",
    r"\bkanye\b",
    r"\brogan\b",
    r"\bmorgan\s+freeman\b",
    r"\bcelebrity\b",
    r"\bdeepfake\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PUBLIC_FIGURE_PATTERNS]


class LicensingPolicyViolation(Exception):
    """Raised when a model or attestation violates licensing or legal rules."""


class LicenseService:
    """Evaluates and enforces licensing boundaries and voice consent policies."""

    @staticmethod
    def is_commercial_release_mode() -> bool:
        """Determines if the application is running in commercial release mode."""
        return os.environ.get("HTH_COMMERCIAL_RELEASE", "0").lower() in ("1", "true", "yes")

    @classmethod
    def check_public_figure_likeness(cls, text: str) -> bool:
        """Check if model name, tags, or description suggest a public figure or celebrity."""
        for pattern in COMPILED_PATTERNS:
            if pattern.search(text):
                return True
        return False

    @classmethod
    def validate_attestation(cls, request: ConsentAttestationRequest, model_name: str = "") -> None:
        """Validate user attestation checklist. Raises LicensingPolicyViolation if invalid."""
        if not request.attested_by or not request.attested_by.strip():
            raise LicensingPolicyViolation("Attestation must include a valid name or identifier.")

        if not request.license_accepted:
            raise LicensingPolicyViolation(
                "You must accept the underlying model license terms to proceed."
            )

        if not (
            request.is_own_voice or request.has_written_consent or request.has_commercial_rights
        ):
            raise LicensingPolicyViolation(
                "You must attest that the voice likeness is your own, backed by explicit written consent, "
                "or licensed with commercial rights."
            )

        if not request.no_celebrity_likeness:
            raise LicensingPolicyViolation(
                "Attestation requires certification that this model does NOT replicate a celebrity or public figure."
            )

        if cls.check_public_figure_likeness(model_name) or cls.check_public_figure_likeness(
            request.notes
        ):
            raise LicensingPolicyViolation(
                "Public figure or celebrity likeness detected. Voice conversion replicating unauthorized "
                "public figures is strictly prohibited by policy."
            )

    @classmethod
    def check_activation_eligibility(
        cls, model: ModelRecord, commercial_mode: bool | None = None
    ) -> None:
        """Verify if a model can legally and safely be activated.

        Raises LicensingPolicyViolation if activation is blocked.
        """
        if commercial_mode is None:
            commercial_mode = cls.is_commercial_release_mode()

        # Public figure block
        if model.is_public_figure_detected:
            raise LicensingPolicyViolation(
                f"Model '{model.name}' is flagged as unauthorized public figure likeness and cannot be activated."
            )

        # Commercial mode restrictions
        if commercial_mode:
            # Block JVS and academic non-commercial models
            if model.license_identifier == "JVS-NonCommercial" or not model.commercial_eligible:
                raise LicensingPolicyViolation(
                    f"Model '{model.name}' has license '{model.license_identifier}' and is blocked from commercial release builds. "
                    "Commercial mode strictly forbids non-commercial academic models."
                )

            # Block Beatrice engine unless formal commercial permission is recorded
            if model.engine_id == "beatrice":
                has_beatrice_commercial_waiver = os.environ.get(
                    "HTH_BEATRICE_COMMERCIAL_AUTH", "0"
                ) in ("1", "true")
                if not has_beatrice_commercial_waiver:
                    raise LicensingPolicyViolation(
                        "Beatrice DSP engine contains proprietary VST inference libraries (contributors.txt:60). "
                        "Commercial activation is blocked without documented commercial rights."
                    )
