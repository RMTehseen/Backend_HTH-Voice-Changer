"""MeanVC2 adapter configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Default model directory: %LOCALAPPDATA%\HTH\MeanVC2Models
def _default_model_dir() -> str:
    local_app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return str(Path(local_app_data) / "HTH" / "MeanVC2Models")

def _default_source_dir() -> str:
    env_val = os.environ.get("HTH_MEANVC2_SOURCE_DIR", "").strip()
    if env_val:
        return env_val
    # MeanVC2 source is extracted to <project_root>/MeanVC2-main (or CWD/MeanVC2-main)
    cwd_candidate = Path.cwd() / "MeanVC2-main"
    if cwd_candidate.is_dir():
        return str(cwd_candidate)
    here = Path(__file__).resolve()
    # Walk up to find project root (has MeanVC2-main folder)
    for parent in here.parents:
        candidate = parent / "MeanVC2-main"
        if candidate.is_dir():
            return str(candidate)
    return ""

@dataclass
class MeanVC2Config:
    """Configuration for the MeanVC2 neural voice conversion adapter."""
    # Compute device
    device: str = "cpu"  # "cpu" or "cuda"
    # Model variant: "40ms" (lower latency) or "120ms" (higher quality)
    model_variant: str = "40ms"
    # Model directory (overridable via HTH_MEANVC2_MODEL_DIR env var)
    model_dir: str = field(default_factory=lambda: os.environ.get("HTH_MEANVC2_MODEL_DIR", _default_model_dir()))
    # MeanVC2 source code directory
    source_dir: str = field(default_factory=_default_source_dir)
    # Worker process heartbeat timeout (seconds)
    heartbeat_timeout_s: float = 10.0
    # Maximum input ring buffer depth (chunks)
    max_input_queue: int = 8
    # Maximum output ring buffer depth (chunks)
    max_output_queue: int = 8
    # Warmup passes before enabling output
    warmup_passes: int = 3

    @property
    def vc_model_path(self) -> str:
        fname = "meanvc2_40ms_40ms.safetensors" if self.model_variant == "40ms" else "meanvc2_120ms_40ms.safetensors"
        return str(Path(self.model_dir) / "ckpts" / "pretrained_models" / fname)

    @property
    def vc_config_path(self) -> str:
        fname = "config_40ms_40ms.json" if self.model_variant == "40ms" else "config_120ms_40ms.json"
        # Config is in source dir
        return str(Path(self.source_dir) / "src" / "config" / fname)

    @property
    def asr_model_path(self) -> str:
        fname = "fastu2pp_80ms.pt" if self.model_variant == "40ms" else "fastu2pp_160ms.pt"
        return str(Path(self.model_dir) / "preprocess" / "ckpts" / fname)

    @property
    def vocoder_path(self) -> str:
        return str(Path(self.model_dir) / "ckpts" / "vocos" / "vocos.pt")

    @property
    def speaker_model_path(self) -> str:
        return str(Path(self.model_dir) / "preprocess" / "ckpts" / "wavlm_large_finetune.pth")

    @property
    def wavlm_config_path(self) -> str:
        return str(Path(self.model_dir) / "preprocess" / "ckpts" / "wavlm_large_cfg.pt")

    def check_assets(self) -> dict[str, bool]:
        """Return dict of asset_name -> exists."""
        has_source = bool(self.source_dir) and Path(self.source_dir).is_dir()
        return {
            "vc_model": Path(self.vc_model_path).is_file(),
            "vc_config": has_source and Path(self.vc_config_path).is_file(),
            "asr_model": Path(self.asr_model_path).is_file(),
            "vocoder": Path(self.vocoder_path).is_file(),
            "speaker_model": Path(self.speaker_model_path).is_file(),
            "wavlm_config": Path(self.wavlm_config_path).is_file(),
            "source_dir": has_source,
        }

    def all_assets_ready(self) -> bool:
        return all(self.check_assets().values())
