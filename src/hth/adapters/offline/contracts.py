"""Contracts and protocol definitions for offline voice-note conversion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from hth.domain.api_contracts import (
    AudioContainerFormat,
    SupportedModelFormat,
)


@dataclass(frozen=True)
class AudioMetadata:
    """Validated metadata of an input audio file."""

    container_format: AudioContainerFormat
    duration_seconds: float
    file_size_bytes: int
    sample_rate: int
    channels: int
    magic_validated: bool


@runtime_checkable
class OfflineConverterProtocol(Protocol):
    """Protocol for offline voice conversion implementations."""

    @property
    def converter_id(self) -> str:
        """Unique identifier for this converter adapter."""
        ...

    @property
    def supported_model_formats(self) -> set[SupportedModelFormat]:
        """Set of model formats supported by this converter."""
        ...

    def validate_model(self, model_id: str, model_format: SupportedModelFormat) -> None:
        """Validate that the model format is compatible with this engine.

        Raises IncompatibleModelError if format does not match.
        """
        ...

    async def convert(
        self,
        job_id: str,
        input_file: Path,
        output_file: Path,
        target_format: AudioContainerFormat,
        model_id: str,
        pitch_shift: float,
        formant_shift: float,
        cancel_token: asyncio.Event,
        progress_callback: Callable[[float, int, int], None],
    ) -> Path:
        """Process input file in bounded chunks, reporting progress and honoring cancellation.

        Args:
            job_id: Unique job identifier.
            input_file: Safe absolute path to input audio file.
            output_file: Safe absolute path to target output file.
            target_format: Requested output container format.
            model_id: ID of the voice model to apply.
            pitch_shift: Semitone pitch shift value.
            formant_shift: Formant shift value.
            cancel_token: Event monitored during processing; if set, must halt immediately.
            progress_callback: Callback invoked on each chunk with (progress_0_to_1, processed_frames, total_frames).

        Returns:
            Path to the completed output file.

        Raises:
            JobCancelledError: If cancellation token was set.
            AudioFormatError: If requested codec is missing or format conversion fails.
            IncompatibleModelError: If model format is not supported.
            EngineError: For general conversion failures.
        """
        ...
