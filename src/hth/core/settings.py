"""HTH application settings loaded from environment / .env file."""

import os
import secrets
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_db_path() -> str:
    local_app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return os.path.join(local_app_data, "HTH", "hth.db")


def _default_log_dir() -> str:
    local_app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return os.path.join(local_app_data, "HTH", "logs")


def _default_voice_notes_dir() -> str:
    local_app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return os.path.join(local_app_data, "HTH", "voice_notes")


def _default_env_files() -> tuple[str, ...]:
    """Find .env in cwd or backend folder for reliable discovery."""
    backend_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
    return (".env", backend_env)


class Settings(BaseSettings):
    host: str = "127.0.0.1"  # NEVER 0.0.0.0
    port: int = 8765
    db_path: str = ""
    log_dir: str = ""
    voice_notes_dir: str = ""
    max_file_size_mb: int = 25
    max_duration_seconds: int = 300
    retention_hours: int = 24
    ffmpeg_path: str = ""
    session_token: str = ""
    log_level: str = "INFO"
    engine_adapter: str = "dsp"  # "dsp" | "pipeline" | "fake" | "beatrice" | "applio" | "meanvc2"
    meanvc2_device: str = "cpu"  # "cpu" or "cuda"
    meanvc2_variant: str = "40ms"  # "40ms" or "120ms"
    meanvc2_model_dir: str = ""  # override model directory
    commercial_build: bool = True  # Enforces exclusion of proprietary/non-commercial engines
    single_instance: bool = True
    reload: bool = False

    model_config = SettingsConfigDict(
        env_prefix="HTH_",
        env_file=_default_env_files(),
        extra="ignore",
    )

    def model_post_init(self, __context: object) -> None:
        if self.host not in ("127.0.0.1", "localhost"):
            raise ValueError(
                f"Prohibited host binding: {self.host!r}. API must bind to 127.0.0.1 only."
            )
        if not self.db_path:
            object.__setattr__(self, "db_path", _default_db_path())
        if not self.log_dir:
            object.__setattr__(self, "log_dir", _default_log_dir())
        if not self.voice_notes_dir:
            object.__setattr__(self, "voice_notes_dir", _default_voice_notes_dir())
        if not self.session_token:
            object.__setattr__(self, "session_token", secrets.token_urlsafe(32))


# NOTE: @lru_cache caches the Settings singleton across test runs.
# In tests, call get_settings.cache_clear() in fixtures or use
# app.dependency_overrides to inject a test-specific Settings instance.
@lru_cache
def get_settings() -> Settings:
    return Settings()
