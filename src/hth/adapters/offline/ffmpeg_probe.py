"""ffmpeg detection and capability probing without shell execution.

Enforces Rule S-06: Never use subprocess with shell=True. Always pass argument lists.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from hth.core.logging import get_logger

logger = get_logger("hth.adapters.offline.ffmpeg")


@dataclass(frozen=True)
class FfmpegCapabilities:
    """Capabilities discovered from the local ffmpeg installation."""

    detected: bool
    version: str | None = None
    binary_path: str | None = None
    encoders: set[str] = field(default_factory=set)

    @property
    def supports_mp3(self) -> bool:
        return "libmp3lame" in self.encoders or "mp3" in self.encoders

    @property
    def supports_ogg_opus(self) -> bool:
        return "libopus" in self.encoders or "opus" in self.encoders or "vorbis" in self.encoders

    @property
    def supports_aac(self) -> bool:
        return "aac" in self.encoders or "libfdk_aac" in self.encoders

    @property
    def supported_output_formats(self) -> list[str]:
        formats = ["wav"]  # WAV PCM is always supported natively
        if self.supports_mp3:
            formats.append("mp3")
        if self.supports_ogg_opus:
            formats.append("ogg")
        if self.supports_aac:
            formats.append("m4a")
        return formats


def probe_ffmpeg(custom_path: str = "") -> FfmpegCapabilities:
    """Probe the system for ffmpeg and its compiled audio encoders.

    Checks:
    1. Explicit custom_path if provided.
    2. %LOCALAPPDATA%\\HTH\\bin\\ffmpeg.exe
    3. System PATH via shutil.which("ffmpeg")

    Never executes with shell=True.
    """
    candidate_path: str | None = None

    if custom_path and os.path.isfile(custom_path):
        candidate_path = custom_path
    else:
        # Check jailed local appdata bin
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            jailed_ffmpeg = Path(local_app_data) / "HTH" / "bin" / "ffmpeg.exe"
            if jailed_ffmpeg.is_file():
                candidate_path = str(jailed_ffmpeg)

    if not candidate_path:
        found_in_path = shutil.which("ffmpeg")
        if found_in_path:
            candidate_path = found_in_path

    if not candidate_path:
        logger.info("ffmpeg not detected on system — falling back to native WAV only")
        return FfmpegCapabilities(detected=False)

    # Probe version
    version_str: str | None = None
    try:
        proc = subprocess.run(
            [candidate_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5.0,
            shell=False,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            first_line = proc.stdout.splitlines()[0]
            version_str = first_line.strip()
    except Exception as exc:
        logger.warning("Failed to execute ffmpeg -version", error=str(exc))
        return FfmpegCapabilities(detected=False)

    # Probe encoders
    encoders: set[str] = set()
    try:
        proc_enc = subprocess.run(
            [candidate_path, "-encoders"],
            capture_output=True,
            text=True,
            timeout=5.0,
            shell=False,
            check=False,
        )
        if proc_enc.returncode == 0 and proc_enc.stdout:
            for line in proc_enc.stdout.splitlines():
                line_stripped = line.strip()
                # Audio encoders start with A (e.g. "A..... libmp3lame")
                if line_stripped.startswith("A"):
                    parts = line_stripped.split()
                    if len(parts) >= 2:
                        encoder_name = parts[1].strip()
                        encoders.add(encoder_name)
    except Exception as exc:
        logger.warning("Failed to query ffmpeg -encoders", error=str(exc))

    logger.info(
        "ffmpeg detected",
        binary_path=candidate_path,
        version=version_str,
        encoders_count=len(encoders),
    )

    return FfmpegCapabilities(
        detected=True,
        version=version_str,
        binary_path=candidate_path,
        encoders=encoders,
    )
