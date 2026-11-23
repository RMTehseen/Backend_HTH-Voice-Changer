"""VoiceNoteService — orchestration for live-route guide and bounded offline conversion.

Enforces:
- Mutual exclusion between offline conversion and live engine streaming (Rule G-04 / CPU protection)
- Jailed local storage under %LOCALAPPDATA%\\HTH\\voice_notes (Rule S-05 / Rule G-02)
- Zero external uploads (Rule G-02)
- Strict model format rejection (Applio vs Beatrice isolation)
- Cooperative cancellation with immediate temp file cleanup
- 24-hour retention sweep
"""

from __future__ import annotations

import asyncio
import datetime
import shutil
import uuid
from pathlib import Path
from typing import Optional

from hth.adapters.offline.audio_validator import validate_audio_file
from hth.adapters.offline.contracts import OfflineConverterProtocol
from hth.adapters.offline.ffmpeg_probe import probe_ffmpeg
from hth.adapters.offline.production_converter import ProductionOfflineVoiceConverter
from hth.core.logging import get_logger
from hth.core.settings import Settings, get_settings
from hth.domain.api_contracts import (
    AudioContainerFormat,
    EngineState,
    LiveRouteGuideResponse,
    RetentionCleanupResponse,
    VoiceNoteCapabilities,
    VoiceNoteJobCreate,
    VoiceNoteJobStatus,
    VoiceNoteJobStatusEnum,
)
from hth.domain.errors import (
    JobCancelledError,
    ResourceConflictError,
)
from hth.services.engine_service import EngineService

logger = get_logger("hth.services.voice_notes")


class VoiceNoteService:
    """Orchestrates voice-note live routing guidance and offline file conversion."""

    def __init__(
        self,
        engine_service: EngineService,
        converter: Optional[OfflineConverterProtocol] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._engine_service = engine_service
        self._converter = converter or ProductionOfflineVoiceConverter()
        self._settings = settings or get_settings()

        self._base_dir = Path(self._settings.voice_notes_dir).resolve()
        self._temp_dir = self._base_dir / "temp"
        self._exports_dir = self._base_dir / "exports"

        # Ensure directories exist locally
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._exports_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, VoiceNoteJobStatus] = {}
        self._cancel_tokens: dict[str, asyncio.Event] = {}
        self._input_file_paths: dict[str, Path] = {}
        self._output_file_paths: dict[str, Path] = {}

        self._current_active_job_id: str | None = None
        self._lock = asyncio.Lock()

    def is_offline_active(self) -> bool:
        """Query whether an offline conversion job is actively executing."""
        return self._current_active_job_id is not None

    async def get_live_guide(self) -> LiveRouteGuideResponse:
        """Generate dynamic live routing instructions for WhatsApp Desktop."""
        health = await self._engine_service.get_health()
        live_running = health.state in (EngineState.RUNNING, EngineState.STARTING)

        # Query devices to check for VB-CABLE
        devices = await self._engine_service._engine.list_devices()
        vb_cable_installed = any("cable" in d.friendly_name.lower() for d in devices)

        current_out = None
        if (
            hasattr(self._engine_service._engine, "_active_config")
            and self._engine_service._engine._active_config
        ):
            current_out = str(self._engine_service._engine._active_config.output_device_id)

        steps = [
            "1. Confirm VB-CABLE driver is installed ('CABLE Input' and 'CABLE Output' visible in Windows Sound Settings).",
            "2. In HTH Voice Changer, select 'CABLE Input' as the Target Output Device.",
            "3. Start HTH Voice Conversion (verify persistent '🔴 AI Voice Active' indicator is shown).",
            "4. Open WhatsApp Desktop -> Settings -> Audio/Video (or Windows 11 App volume preferences) and set Microphone to 'CABLE Output (VB-Audio Virtual Cable)'.",
            "5. In your WhatsApp chat, press and hold the native microphone button to record your voice note.",
            "6. Release to send. WhatsApp records directly from the virtual mic, generating a genuine native voice-note bubble with waveform.",
        ]

        notice = (
            "WhatsApp Desktop records audio directly from its configured input device. "
            "Routing converted audio through CABLE Output produces an authentic WhatsApp voice-note bubble (PTT) with native waveform display."
        )

        warning = (
            "IMPORTANT: Pre-recorded audio files converted via offline processing (Path B) will be sent as "
            "file attachments/documents in WhatsApp Desktop, NOT as native voice-note bubbles. "
            "To send a genuine voice-note bubble, use this live-recording route (Path A)."
        )

        return LiveRouteGuideResponse(
            vb_cable_installed=vb_cable_installed,
            live_engine_running=live_running,
            current_output_device=current_out,
            recommended_whatsapp_input="CABLE Output (VB-Audio Virtual Cable)",
            steps=steps,
            whatsapp_bubble_notice=notice,
            attachment_warning=warning,
        )

    async def get_capabilities(self) -> VoiceNoteCapabilities:
        """Query local ffmpeg capabilities and service constraints."""
        ffmpeg_caps = probe_ffmpeg(self._settings.ffmpeg_path)
        health = await self._engine_service.get_health()
        live_running = health.state in (EngineState.RUNNING, EngineState.STARTING)

        return VoiceNoteCapabilities(
            ffmpeg_detected=ffmpeg_caps.detected,
            ffmpeg_version=ffmpeg_caps.version,
            available_encoders=sorted(list(ffmpeg_caps.encoders)),
            supported_input_formats=["wav", "mp3", "ogg", "flac", "m4a"],
            supported_output_formats=ffmpeg_caps.supported_output_formats,
            max_file_size_mb=self._settings.max_file_size_mb,
            max_duration_seconds=self._settings.max_duration_seconds,
            live_conversion_active=live_running,
            offline_job_active=self.is_offline_active(),
        )

    async def submit_job(
        self,
        file_bytes: bytes,
        filename: str,
        job_params: VoiceNoteJobCreate,
    ) -> VoiceNoteJobStatus:
        """Submit a file for offline conversion, validating resources, bounds, and model format."""
        # 1. Concurrency Check: Mutual exclusion with live engine
        health = await self._engine_service.get_health()
        if health.state in (EngineState.RUNNING, EngineState.STARTING):
            raise ResourceConflictError(
                "Cannot start offline conversion while live voice conversion is running. "
                "Please stop live conversion first to prevent CPU contention and audio dropouts.",
                competing_task="live_engine_stream",
            )

        # 2. Concurrency Check: Mutual exclusion with concurrent offline jobs
        async with self._lock:
            if self._current_active_job_id is not None:
                raise ResourceConflictError(
                    f"Another offline conversion job ({self._current_active_job_id}) is currently processing. "
                    "Simultaneous offline jobs are prohibited to preserve system resources.",
                    competing_task=f"job_{self._current_active_job_id}",
                )

            # 3. Model format compatibility validation (strict provider isolation)
            self._converter.validate_model(job_params.model_id, job_params.model_format)

            # 4. Input file validation: magic bytes, size limits, duration bounds
            max_bytes = self._settings.max_file_size_mb * 1024 * 1024
            metadata = validate_audio_file(
                file_bytes=file_bytes,
                original_filename=filename,
                max_size_bytes=max_bytes,
                max_duration_seconds=float(self._settings.max_duration_seconds),
            )

            job_id = str(uuid.uuid4())
            now = datetime.datetime.now(datetime.timezone.utc)

            # 5. Jail paths under local app data
            job_temp_dir = self._temp_dir / job_id
            job_temp_dir.mkdir(parents=True, exist_ok=True)

            input_suffix = Path(filename).suffix.lower()
            input_file = job_temp_dir / f"source{input_suffix}"
            input_file.write_bytes(file_bytes)

            output_file = self._exports_dir / f"{job_id}_converted.{job_params.target_format.value}"

            job_status = VoiceNoteJobStatus(
                job_id=job_id,
                status=VoiceNoteJobStatusEnum.PENDING,
                progress=0.0,
                input_filename=filename,
                input_format=metadata.container_format.value,
                output_format=job_params.target_format.value,
                duration_seconds=metadata.duration_seconds,
                processed_frames=0,
                total_frames=0,
                has_source_audio=True,
                has_converted_audio=False,
                created_at=now,
            )

            self._jobs[job_id] = job_status
            self._cancel_tokens[job_id] = asyncio.Event()
            self._input_file_paths[job_id] = input_file
            self._output_file_paths[job_id] = output_file
            self._current_active_job_id = job_id

            # 6. Spawn bounded background worker
            asyncio.create_task(
                self._run_conversion_worker(
                    job_id=job_id,
                    input_file=input_file,
                    output_file=output_file,
                    target_format=job_params.target_format,
                    model_id=job_params.model_id,
                    pitch_shift=job_params.pitch_shift,
                    formant_shift=job_params.formant_shift,
                )
            )

            return job_status

    async def _run_conversion_worker(
        self,
        job_id: str,
        input_file: Path,
        output_file: Path,
        target_format: AudioContainerFormat,
        model_id: str,
        pitch_shift: float,
        formant_shift: float,
    ) -> None:
        """Background chunked conversion worker with cooperative cancellation."""
        job = self._jobs[job_id]
        job.status = VoiceNoteJobStatusEnum.PROCESSING
        cancel_token = self._cancel_tokens[job_id]

        def on_progress(progress: float, processed_frames: int, total_frames: int) -> None:
            job.progress = progress
            job.processed_frames = processed_frames
            job.total_frames = total_frames

        try:
            await self._converter.convert(
                job_id=job_id,
                input_file=input_file,
                output_file=output_file,
                target_format=target_format,
                model_id=model_id,
                pitch_shift=pitch_shift,
                formant_shift=formant_shift,
                cancel_token=cancel_token,
                progress_callback=on_progress,
            )

            job.status = VoiceNoteJobStatusEnum.COMPLETED
            job.progress = 1.0
            job.has_converted_audio = output_file.is_file()
            job.completed_at = datetime.datetime.now(datetime.timezone.utc)
            logger.info("Voice note conversion completed successfully", job_id=job_id)

        except JobCancelledError as exc:
            job.status = VoiceNoteJobStatusEnum.CANCELLED
            job.error = str(exc)
            job.completed_at = datetime.datetime.now(datetime.timezone.utc)
            self._cleanup_job_temp(job_id)
            if output_file.exists():
                output_file.unlink()
            logger.info("Voice note conversion job cancelled and cleaned up", job_id=job_id)

        except Exception as exc:
            job.status = VoiceNoteJobStatusEnum.FAILED
            job.error = str(exc)
            job.completed_at = datetime.datetime.now(datetime.timezone.utc)
            self._cleanup_job_temp(job_id)
            logger.error("Voice note conversion job failed", job_id=job_id, error=str(exc))

        finally:
            # Free active offline job concurrency lock
            async with self._lock:
                if self._current_active_job_id == job_id:
                    self._current_active_job_id = None

    async def cancel_job(self, job_id: str) -> VoiceNoteJobStatus:
        """Cancel an ongoing offline conversion job and delete temporary files immediately."""
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Job '{job_id}' not found.")

        if job.status in (
            VoiceNoteJobStatusEnum.COMPLETED,
            VoiceNoteJobStatusEnum.CANCELLED,
            VoiceNoteJobStatusEnum.FAILED,
        ):
            return job

        token = self._cancel_tokens.get(job_id)
        if token:
            token.set()

        job.status = VoiceNoteJobStatusEnum.CANCELLED
        job.error = "Cancelled by user"
        job.completed_at = datetime.datetime.now(datetime.timezone.utc)

        # Clean temp directory immediately
        self._cleanup_job_temp(job_id)

        out_path = self._output_file_paths.get(job_id)
        if out_path and out_path.exists():
            out_path.unlink()

        async with self._lock:
            if self._current_active_job_id == job_id:
                self._current_active_job_id = None

        return job

    def get_job_status(self, job_id: str) -> VoiceNoteJobStatus:
        """Fetch status snapshot of a conversion job."""
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Job '{job_id}' not found.")
        return job

    def get_source_audio_path(self, job_id: str) -> Path:
        """Retrieve path to original uploaded audio for A/B preview."""
        path = self._input_file_paths.get(job_id)
        if not path or not path.is_file():
            raise FileNotFoundError(f"Source audio file not found for job '{job_id}'.")
        return path

    def get_converted_audio_path(self, job_id: str) -> Path:
        """Retrieve path to converted audio for A/B preview."""
        path = self._output_file_paths.get(job_id)
        if not path or not path.is_file():
            raise FileNotFoundError(f"Converted audio file not found for job '{job_id}'.")
        return path

    def _cleanup_job_temp(self, job_id: str) -> None:
        """Remove intermediate temporary files for a specific job."""
        job_temp = self._temp_dir / job_id
        if job_temp.exists():
            try:
                shutil.rmtree(job_temp)
            except Exception as exc:
                logger.warning("Failed to clean up job temp dir", job_id=job_id, error=str(exc))

    def cleanup_retention(self, retention_hours: int = 24) -> RetentionCleanupResponse:
        """Purge temporary files and exports older than the retention threshold."""
        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff = now - datetime.timedelta(hours=retention_hours)

        deleted_temp = 0
        deleted_exports = 0
        freed_bytes = 0

        # Sweep temp dir for orphaned folders
        if self._temp_dir.exists():
            for entry in self._temp_dir.iterdir():
                if entry.is_dir() and entry.name != self._current_active_job_id:
                    try:
                        mtime = datetime.datetime.fromtimestamp(
                            entry.stat().st_mtime, tz=datetime.timezone.utc
                        )
                        if mtime < cutoff:
                            dir_size = sum(
                                f.stat().st_size for f in entry.rglob("*") if f.is_file()
                            )
                            shutil.rmtree(entry)
                            deleted_temp += 1
                            freed_bytes += dir_size
                    except Exception:
                        pass

        # Sweep exports dir
        if self._exports_dir.exists():
            for entry in self._exports_dir.iterdir():
                if entry.is_file():
                    try:
                        mtime = datetime.datetime.fromtimestamp(
                            entry.stat().st_mtime, tz=datetime.timezone.utc
                        )
                        if mtime < cutoff:
                            size = entry.stat().st_size
                            entry.unlink()
                            deleted_exports += 1
                            freed_bytes += size
                    except Exception:
                        pass

        return RetentionCleanupResponse(
            deleted_temp_files=deleted_temp,
            deleted_export_files=deleted_exports,
            freed_bytes=freed_bytes,
            timestamp=now,
        )
