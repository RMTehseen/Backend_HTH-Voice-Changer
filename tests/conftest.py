"""Pytest configuration and shared test fixtures."""

import multiprocessing as mp
import os
import sys
from typing import Generator, Tuple

import pytest

# Ensure backend/src is on the path for test execution
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

os.environ["HTH_USE_FAKE_AUDIO_DEVICES"] = "1"
os.environ["HTH_ENGINE_ADAPTER"] = "fake"

# ---------------------------------------------------------------------------
# Phase 1 fixtures (kept for existing Beatrice adapter tests)
# ---------------------------------------------------------------------------
from hth.adapters.ipc.connection import IPCConnection  # noqa: E402
from hth.domain.contracts import EngineParams, LiveConfig  # noqa: E402


@pytest.fixture
def test_token() -> str:
    """Provides a deterministic test session token."""
    return "test-session-token-32-chars-long-abc123"


@pytest.fixture
def mock_pipe_pair() -> Generator[Tuple[IPCConnection, IPCConnection], None, None]:
    """Provides a connected pair of IPCConnection instances over a multiprocessing Pipe."""
    p1, p2 = mp.Pipe(duplex=True)
    c1 = IPCConnection(p1)
    c2 = IPCConnection(p2)
    try:
        yield c1, c2
    finally:
        c1.close()
        c2.close()


@pytest.fixture
def default_config() -> LiveConfig:
    """Standard live configuration for testing."""
    return LiveConfig(
        input_device_id=0,
        output_device_id=1,
        sample_rate=16000,
        block_size=160,
        model_id="jvs",
        params=EngineParams(
            pitch_shift=2.0,
            formant_shift=1.0,
            volume=0.8,
            gate_threshold=0.01,
            hear_yourself=False,
            muted=False,
        ),
    )


# ---------------------------------------------------------------------------
# Phase 2 fixtures
# ---------------------------------------------------------------------------
from hth.core.settings import Settings, get_settings  # noqa: E402


@pytest.fixture
def test_settings() -> Settings:
    """Return a fresh Settings instance with a fixed token for testing."""
    get_settings.cache_clear()
    s = Settings(
        host="127.0.0.1",
        port=8765,
        db_path=":memory:",
        session_token="test-token-fixed-32-chars-abc12345",
        log_level="DEBUG",
        engine_adapter="fake",
    )
    return s
