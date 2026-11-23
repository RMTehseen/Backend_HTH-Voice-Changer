"""Beatrice V2 Voice Engine Adapter Package (PERSONAL_POC_ONLY)."""

from hth.adapters.beatrice.adapter import BeatriceAdapter
from hth.adapters.beatrice.capabilities import BEATRICE_CAPABILITIES
from hth.adapters.beatrice.supervisor import BeatriceProcessSupervisor

# PERSONAL_POC_ONLY — do not enable in commercial build
# Proprietary VST engine: personal/non-commercial restrictions (contributors.txt:60)
# JVS Corpus models: non-commercial (beatrice_paraphernalia_jvs.toml:7)
PERSONAL_POC_ONLY: bool = True
COMMERCIAL_DISTRIBUTION_ENABLED: bool = False

__all__ = [
    "PERSONAL_POC_ONLY",
    "COMMERCIAL_DISTRIBUTION_ENABLED",
    "BeatriceAdapter",
    "BeatriceProcessSupervisor",
    "BEATRICE_CAPABILITIES",
]
