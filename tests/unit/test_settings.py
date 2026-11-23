"""Unit tests for Settings."""

import pytest

from hth.core.settings import Settings, get_settings


def test_default_host() -> None:
    get_settings.cache_clear()
    s = Settings(session_token="test-token-fixed-32-chars-abc12345")
    assert s.host == "127.0.0.1"
    assert s.port == 8765


def test_session_token_auto_generated() -> None:
    get_settings.cache_clear()
    s = Settings()
    assert len(s.session_token) > 0


def test_db_path_uses_localappdata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", "C:\\TestAppData")
    get_settings.cache_clear()
    s = Settings(session_token="tok")
    assert "HTH" in s.db_path
    assert "hth.db" in s.db_path
    get_settings.cache_clear()
