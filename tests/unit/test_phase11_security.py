"""Unit tests for Phase 11: Security Hardening & Release Verification."""

import os
import tempfile

import pytest
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Scope

from hth.core.database import (
    create_pre_migration_backup,
    restore_database_from_backup,
    verify_database_integrity,
)
from hth.core.logging import _redact_sensitive
from hth.core.process_job import SingleInstanceLock
from hth.core.security import SecurityMiddleware, verify_session_token
from hth.core.settings import Settings, get_settings


def test_settings_host_validation() -> None:
    """Rule S-01: Prohibit non-loopback host bindings."""
    s_valid = Settings(host="127.0.0.1")
    assert s_valid.host == "127.0.0.1"

    with pytest.raises(ValueError, match="Prohibited host binding"):
        Settings(host="0.0.0.0")

    with pytest.raises(ValueError, match="Prohibited host binding"):
        Settings(host="192.168.1.50")


def test_verify_session_token() -> None:
    """Rule S-02: Constant-time verification of session tokens."""
    settings = get_settings()
    token = settings.session_token
    assert verify_session_token(token) is True
    assert verify_session_token("invalid-token") is False
    assert verify_session_token("") is False


@pytest.mark.asyncio
async def test_security_middleware_host_validation() -> None:
    """Rule S-03: Validate Host header to prevent DNS rebinding."""
    middleware = SecurityMiddleware(app=None)  # type: ignore[arg-type]

    async def mock_call_next(req: Request) -> Response:
        return Response("OK", status_code=200)

    # Valid Host
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/test",
        "headers": [
            (b"host", b"127.0.0.1:8765"),
            (b"x-hth-token", get_settings().session_token.encode("utf-8")),
        ],
    }
    req = Request(scope)
    res = await middleware.dispatch(req, mock_call_next)
    assert res.status_code == 200

    # Malicious Host (DNS rebinding)
    scope["headers"] = [
        (b"host", b"attacker.com"),
        (b"x-hth-token", get_settings().session_token.encode("utf-8")),
    ]
    req = Request(scope)
    res = await middleware.dispatch(req, mock_call_next)
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_security_middleware_origin_validation() -> None:
    """Rule S-03: Restrict Origin header to local loopback and Tauri custom protocols."""
    middleware = SecurityMiddleware(app=None)  # type: ignore[arg-type]

    async def mock_call_next(req: Request) -> Response:
        return Response("OK", status_code=200)

    # Valid Origin: http://tauri.localhost (Windows Tauri 2 default)
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/test",
        "headers": [
            (b"host", b"127.0.0.1:8765"),
            (b"origin", b"http://tauri.localhost"),
            (b"x-hth-token", get_settings().session_token.encode("utf-8")),
        ],
    }
    req = Request(scope)
    res = await middleware.dispatch(req, mock_call_next)
    assert res.status_code == 200

    # Invalid Origin: https://evil.com
    scope["headers"] = [
        (b"host", b"127.0.0.1:8765"),
        (b"origin", b"https://evil.com"),
        (b"x-hth-token", get_settings().session_token.encode("utf-8")),
    ]
    req = Request(scope)
    res = await middleware.dispatch(req, mock_call_next)
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_security_middleware_bearer_token() -> None:
    """Rule S-02: Support Authorization: Bearer <token>."""
    middleware = SecurityMiddleware(app=None)  # type: ignore[arg-type]

    async def mock_call_next(req: Request) -> Response:
        return Response("OK", status_code=200)

    token = get_settings().session_token
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/test",
        "headers": [
            (b"host", b"127.0.0.1:8765"),
            (b"authorization", f"Bearer {token}".encode("utf-8")),
        ],
    }
    req = Request(scope)
    res = await middleware.dispatch(req, mock_call_next)
    assert res.status_code == 200


def test_log_redaction_and_path_sanitization() -> None:
    """Rule S-10: Redact sensitive tokens and sanitize personal user profile paths."""
    user_home = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    test_event = {
        "event": "user_action",
        "token": "secret_session_token_12345",
        "audio_data": [0.1, 0.2, 0.3],
        "model_path": os.path.join(user_home, "AppData", "Local", "HTH", "models", "voice1.bin"),
        "safe_param": "48000Hz",
    }

    sanitized = _redact_sensitive(None, "info", test_event)
    assert "token" not in sanitized
    assert "audio_data" not in sanitized
    assert sanitized["safe_param"] == "48000Hz"
    assert user_home not in sanitized["model_path"]
    assert "[USERPROFILE]" in sanitized["model_path"]


def test_single_instance_lock() -> None:
    """Process single-instance lock file acquisition and release."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = os.path.join(tmpdir, "test.lock")
        lock1 = SingleInstanceLock(lock_path)
        lock2 = SingleInstanceLock(lock_path)

        assert lock1.acquire() is True
        assert lock1.is_locked is True

        # Second lock must fail
        assert lock2.acquire() is False
        assert lock2.is_locked is False

        # Release first lock
        lock1.release()
        assert lock1.is_locked is False

        # Now second lock should succeed
        assert lock2.acquire() is True
        lock2.release()


def test_database_backup_and_rollback() -> None:
    """Database backup creation, integrity verification, and restoration."""
    import sqlite3

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "hth_test.db")
        backup_dir = os.path.join(tmpdir, "backups")

        # Initialize a test sqlite db
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test_data (id INTEGER PRIMARY KEY, msg TEXT);")
        conn.execute("INSERT INTO test_data (msg) VALUES ('original');")
        conn.commit()
        conn.close()

        # Update settings temporarily to point to test db
        settings = get_settings()
        old_db = settings.db_path
        try:
            object.__setattr__(settings, "db_path", db_path)

            assert verify_database_integrity() is True

            backup_file = create_pre_migration_backup(backup_dir)
            assert backup_file is not None
            assert os.path.exists(backup_file)

            # Modify the db
            conn = sqlite3.connect(db_path)
            conn.execute("UPDATE test_data SET msg = 'corrupted_state';")
            conn.commit()
            conn.close()

            # Restore from backup
            restored = restore_database_from_backup(backup_file)
            assert restored is True

            # Check original content restored
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT msg FROM test_data WHERE id = 1;")
            row = cur.fetchone()
            conn.close()
            assert row is not None and row[0] == "original"

        finally:
            object.__setattr__(settings, "db_path", old_db)
