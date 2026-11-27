"""Unit and integration tests for secure authentication and bootstrap ticket mechanisms.

Validates:
1. Ephemeral single-use bootstrap ticket generation and atomic invalidation.
2. Ticket expiration enforcement.
3. Single-use WebSocket ticket issuance and redemption.
4. Loopback cookie issuance (HttpOnly, SameSite=Strict).
5. DNS rebinding prevention (Host validation).
6. Cross-origin validation (Origin header checks).
7. Prevention of token query parameter reliance.
"""

import concurrent.futures
import time

import pytest
from httpx import ASGITransport, AsyncClient

from hth.core.app import create_app
from hth.core.security import TicketStore, ticket_store
from hth.core.settings import get_settings


@pytest.fixture
def clean_ticket_store() -> TicketStore:
    return TicketStore()


def test_ticket_store_bootstrap_lifecycle(clean_ticket_store: TicketStore) -> None:
    """Verifies that bootstrap tickets can be redeemed once and only once."""
    ticket = clean_ticket_store.generate_bootstrap_ticket(ttl_seconds=10.0)
    assert ticket
    assert len(ticket) >= 32

    # First redemption succeeds
    assert clean_ticket_store.redeem_bootstrap_ticket(ticket) is True

    # Immediate second redemption fails (atomic single-use)
    assert clean_ticket_store.redeem_bootstrap_ticket(ticket) is False


def test_ticket_store_concurrent_bootstrap_redemption(clean_ticket_store: TicketStore) -> None:
    """Verifies that under simultaneous concurrent access, exactly 1 thread succeeds."""
    ticket = clean_ticket_store.generate_bootstrap_ticket(ttl_seconds=30.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(clean_ticket_store.redeem_bootstrap_ticket, ticket) for _ in range(20)
        ]
        results = [f.result() for f in futures]

    assert results.count(True) == 1
    assert results.count(False) == 19


def test_ticket_store_bootstrap_expiry(clean_ticket_store: TicketStore) -> None:
    """Verifies that expired bootstrap tickets are rejected."""
    ticket = clean_ticket_store.generate_bootstrap_ticket(ttl_seconds=0.01)
    time.sleep(0.05)
    assert clean_ticket_store.redeem_bootstrap_ticket(ticket) is False


def test_ticket_store_ws_ticket_lifecycle(clean_ticket_store: TicketStore) -> None:
    """Verifies single-use WebSocket tickets."""
    ws_ticket = clean_ticket_store.generate_ws_ticket(ttl_seconds=10.0)
    assert ws_ticket

    # First redemption succeeds
    assert clean_ticket_store.redeem_ws_ticket(ws_ticket) is True

    # Second redemption fails
    assert clean_ticket_store.redeem_ws_ticket(ws_ticket) is False


def test_ticket_store_concurrent_ws_redemption(clean_ticket_store: TicketStore) -> None:
    """Verifies that under simultaneous concurrent access, exactly 1 thread succeeds for WS ticket."""
    ws_ticket = clean_ticket_store.generate_ws_ticket(ttl_seconds=30.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(clean_ticket_store.redeem_ws_ticket, ws_ticket) for _ in range(20)
        ]
        results = [f.result() for f in futures]

    assert results.count(True) == 1
    assert results.count(False) == 19


@pytest.mark.asyncio
async def test_api_redeem_ticket_success_and_cookie() -> None:
    """POST /api/v1/auth/redeem-ticket sets HttpOnly cookie and invalidates ticket."""
    app = create_app()
    ticket = ticket_store.generate_bootstrap_ticket(ttl_seconds=30.0)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        # 1. Redeem ticket
        resp = await client.post(
            "/api/v1/auth/redeem-ticket",
            json={"ticket": ticket},
            headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["token"] == get_settings().session_token

        # Check Set-Cookie header contains hth_session with strict attributes (Task 1)
        cookie_header = resp.headers.get("set-cookie", "")
        cookie_lower = cookie_header.lower()
        assert "hth_session=" in cookie_header
        assert "httponly" in cookie_lower
        assert "samesite=strict" in cookie_lower
        assert "path=/" in cookie_lower
        assert "secure" not in cookie_lower
        assert "domain=" not in cookie_lower

        # 2. Replay redemption must fail with 401
        replay_resp = await client.post(
            "/api/v1/auth/redeem-ticket",
            json={"ticket": ticket},
            headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"},
        )
        assert replay_resp.status_code == 401


@pytest.mark.asyncio
async def test_api_cookie_authenticated_request() -> None:
    """Protected endpoints accept requests with valid hth_session cookie."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        # Set cookie directly
        client.cookies.set("hth_session", get_settings().session_token)

        resp = await client.get(
            "/api/v1/auth/check",
            headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "authenticated": True}


@pytest.mark.asyncio
async def test_api_unauthenticated_request_rejected() -> None:
    """Requests to protected endpoints without cookie or token fail with 401."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        resp = await client.get(
            "/api/v1/auth/check",
            headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"},
        )
        assert resp.status_code == 401
        assert resp.json()["code"] == "UNAUTHORIZED"


@pytest.mark.asyncio
async def test_api_dns_rebinding_forbidden() -> None:
    """Requests with external Host headers are rejected with 403 Forbidden."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        resp = await client.get(
            "/api/v1/health",
            headers={"Host": "attacker-dns.com"},
        )
        assert resp.status_code == 403
        assert resp.json()["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_api_strict_cors_and_origin_allowlist() -> None:
    """Verifies that only approved origins are accepted and arbitrary loopback/external origins are rejected."""
    app = create_app()
    transport = ASGITransport(app=app)

    # 1. Approved origins: must succeed with 200 and echo CORS headers
    approved_origins = [
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8765",
        "http://tauri.localhost",  # Windows Tauri 2 default origin (useHttpsScheme not enabled)
    ]
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        for origin in approved_origins:
            resp = await client.get(
                "/api/v1/health",
                headers={"Host": "127.0.0.1:8765", "Origin": origin},
            )
            assert resp.status_code == 200, f"Origin {origin} was unexpectedly rejected"
            assert resp.headers.get("access-control-allow-origin") == origin
            assert resp.headers.get("access-control-allow-credentials") == "true"

    # 2. Disallowed origins: must be rejected with 403 Forbidden
    disallowed_origins = [
        "tauri://localhost",  # Unsupported non-Windows custom protocol scheme
        "https://tauri.localhost",  # Unconfigured HTTPS Tauri scheme must be rejected
        "http://127.0.0.1:9999",  # Arbitrary loopback port
        "http://localhost:5173",  # localhost origin (not canonical)
        "https://127.0.0.1:8765",  # HTTPS loopback
        "https://127.0.0.1:5173",  # HTTPS dev port
        "http://malicious-site.com",  # External domain
        "http://attacker.local",  # Local network attacker
    ]
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        for origin in disallowed_origins:
            resp = await client.get(
                "/api/v1/health",
                headers={"Host": "127.0.0.1:8765", "Origin": origin},
            )
            assert resp.status_code == 403, f"Origin {origin} was unexpectedly allowed"
            assert resp.json()["code"] == "FORBIDDEN"


def test_cors_configuration_never_wildcard_credentials() -> None:
    """Verifies that CORS middleware never configures wildcard origin with allow_credentials."""
    from starlette.middleware.cors import CORSMiddleware

    from hth.core.security import ALLOWED_ORIGINS

    app = create_app()
    cors_mw = None
    for mw in app.user_middleware:
        if mw.cls == CORSMiddleware:
            cors_mw = mw
            break

    assert cors_mw is not None
    assert cors_mw.kwargs.get("allow_credentials") is True
    origins = cors_mw.kwargs.get("allow_origins", [])
    assert "*" not in origins
    assert set(origins) == set(ALLOWED_ORIGINS)


@pytest.mark.asyncio
async def test_cloud_mode_still_enforces_origin_and_session_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """In HTH_CLOUD_MODE=true, tunnel Host varies, but Origin and X-HTH-Token MUST still be enforced."""
    monkeypatch.setenv("HTH_CLOUD_MODE", "true")
    app = create_app()
    transport = ASGITransport(app=app)
    token = get_settings().session_token

    async with AsyncClient(transport=transport, base_url="https://demo-tunnel.trycloudflare.com") as client:
        # 1. Evil origin must be rejected even in cloud mode
        bad_origin = await client.get(
            "/api/v1/auth/check",
            headers={
                "Host": "demo-tunnel.trycloudflare.com",
                "Origin": "https://evil-attacker.example",
                "X-HTH-Token": token,
            },
        )
        assert bad_origin.status_code == 403

        # 2. Missing session token must be rejected with 401 even in cloud mode
        no_tok = await client.get(
            "/api/v1/auth/check",
            headers={
                "Host": "demo-tunnel.trycloudflare.com",
                "Origin": "http://127.0.0.1:5173",
            },
        )
        assert no_tok.status_code == 401

        # 3. Approved origin + valid token succeeds
        ok_resp = await client.get(
            "/api/v1/auth/check",
            headers={
                "Host": "demo-tunnel.trycloudflare.com",
                "Origin": "http://127.0.0.1:5173",
                "X-HTH-Token": token,
            },
        )
        assert ok_resp.status_code == 200
        assert ok_resp.json() == {"ok": True, "authenticated": True}


def test_logging_redaction_sanitizes_ws_ticket() -> None:
    """Task 2 & 3: Verifies robust redaction of tickets/tokens across msgs, args, uvicorn, and tracebacks."""
    import io
    import logging

    from hth.core.logging import TicketRedactionFilter, redact_query_params

    # 1. Verify redact_query_params direct function across raw, token, urlencoded, and ordered params
    secret_ticket = "secret_ws_ticket_xyz9876543210"
    secret_token = "secret_token_abc123456789"
    secret_urlencoded = "%ab%cd%ef%20hex_val"

    # Raw ?ticket= with trailing params
    s1 = redact_query_params(f"GET /api/v1/events/ws?ticket={secret_ticket}&foo=bar HTTP/1.1")
    assert secret_ticket not in s1
    assert "ticket=[REDACTED]" in s1
    assert "foo=bar" in s1

    # Raw ?token= with preceding params
    s2 = redact_query_params(f"GET /api/v1/events/ws?foo=bar&token={secret_token} HTTP/1.1")
    assert secret_token not in s2
    assert "token=[REDACTED]" in s2
    assert "foo=bar" in s2

    # URL-encoded hex values
    s3 = redact_query_params(f"GET /api/v1/events/ws?ticket={secret_urlencoded}&baz=1 HTTP/1.1")
    assert secret_urlencoded not in s3
    assert "ticket=[REDACTED]" in s3
    assert "baz=1" in s3

    # 2. Verify TicketRedactionFilter on logging records (uvicorn, format args, tracebacks)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(TicketRedactionFilter())

    test_logger = logging.getLogger("test.redaction.robust")
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)

    # Uvicorn access log format where path is an argument:
    test_logger.info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:5173",
        "GET",
        f"/api/v1/events/ws?ticket={secret_ticket}&token={secret_token}",
        "1.1",
        101,
    )

    # Application log with format specifier and ticket in format string:
    test_logger.info("Connecting path %s?ticket=%s", "/api/v1/events/ws", "specifier_secret_999")

    # Diagnostic log with single string:
    test_logger.info(
        "Direct log string with ?ticket=%s and urlencoded ?ticket=%s",
        "raw_arg_secret",
        "%ab_url_secret",
    )

    # Exception with traceback containing secret:
    try:
        raise RuntimeError("Connection failed for /ws?ticket=secret_in_exception_traceback")
    except Exception:
        test_logger.error("Error during connection", exc_info=True)

    output = stream.getvalue()

    # Verify no secrets appear anywhere in log output:
    assert secret_ticket not in output
    assert secret_token not in output
    assert "specifier_secret_999" not in output
    assert "raw_arg_secret" not in output
    assert "%ab_url_secret" not in output
    assert "secret_in_exception_traceback" not in output
    assert "ticket=[REDACTED]" in output
    assert "token=[REDACTED]" in output


@pytest.mark.asyncio
async def test_api_ws_ticket_issuance() -> None:
    """POST /api/v1/auth/ws-ticket issues single-use WebSocket ticket."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        client.cookies.set("hth_session", get_settings().session_token)

        resp = await client.post(
            "/api/v1/auth/ws-ticket",
            headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"},
        )
        assert resp.status_code == 200
        ws_ticket = resp.json()["ticket"]
        assert ws_ticket

        # Ticket can be redeemed via ticket_store
        assert ticket_store.redeem_ws_ticket(ws_ticket) is True
        # Cannot be reused
        assert ticket_store.redeem_ws_ticket(ws_ticket) is False


def test_websocket_valid_ticket_with_valid_cookie_consumes_ticket() -> None:
    """1. Valid ticket + valid session cookie -> ticket is consumed and connection succeeds."""
    from starlette.testclient import TestClient

    app = create_app()
    ticket = ticket_store.generate_ws_ticket(ttl_seconds=15.0)
    settings = get_settings()

    with TestClient(app, cookies={"hth_session": settings.session_token}) as client:
        with client.websocket_connect(f"/api/v1/events/ws?ticket={ticket}") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert "session_id" in msg

        # Ticket must have been consumed atomically from store
        assert ticket not in ticket_store._ws_tickets


def test_websocket_ticket_reuse_with_valid_cookie_rejected() -> None:
    """2. Same ticket reused with valid cookie -> connection rejected with code 1008."""
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    app = create_app()
    ticket = ticket_store.generate_ws_ticket(ttl_seconds=15.0)
    settings = get_settings()

    with TestClient(app, cookies={"hth_session": settings.session_token}) as client:
        # First connection consumes ticket
        with client.websocket_connect(f"/api/v1/events/ws?ticket={ticket}") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"

        # Replay with same ticket must be rejected even though valid cookie is present
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(f"/api/v1/events/ws?ticket={ticket}") as ws:
                ws.receive_json()
        assert exc_info.value.code == 1008


def test_websocket_invalid_ticket_with_valid_cookie_rejected() -> None:
    """3. Invalid ticket + valid cookie -> rejected (cookie must not bypass explicit ticket auth)."""
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    app = create_app()
    settings = get_settings()

    with TestClient(app, cookies={"hth_session": settings.session_token}) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/api/v1/events/ws?ticket=invalid-fake-ticket") as ws:
                ws.receive_json()
        assert exc_info.value.code == 1008


def test_websocket_expired_ticket_rejected() -> None:
    """4. Expired ticket -> rejected with code 1008."""
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    app = create_app()
    ticket = ticket_store.generate_ws_ticket(ttl_seconds=0.001)
    time.sleep(0.01)

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(f"/api/v1/events/ws?ticket={ticket}") as ws:
                ws.receive_json()
        assert exc_info.value.code == 1008


def test_websocket_valid_ticket_without_cookie_succeeds() -> None:
    """5. Valid ticket without cookie -> succeeds and consumes ticket."""
    from starlette.testclient import TestClient

    app = create_app()
    ticket = ticket_store.generate_ws_ticket(ttl_seconds=15.0)

    with TestClient(app) as client:
        with client.websocket_connect(f"/api/v1/events/ws?ticket={ticket}") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert "session_id" in msg

        assert ticket not in ticket_store._ws_tickets


def test_websocket_missing_ticket_and_missing_cookie_rejected() -> None:
    """6. Missing ticket and missing cookie -> rejected with code 1008."""
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    app = create_app()
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/api/v1/events/ws") as ws:
                ws.receive_json()
        assert exc_info.value.code == 1008


def test_websocket_legacy_token_query_rejected() -> None:
    """7. Legacy ?token= parameter -> rejected with code 1008."""
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    app = create_app()
    token = get_settings().session_token
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(f"/api/v1/events/ws?token={token}") as ws:
                ws.receive_json()
        assert exc_info.value.code == 1008


def test_websocket_cookie_only_without_ticket_succeeds() -> None:
    """Documented cookie-only path: valid cookie with no ticket parameter succeeds."""
    from starlette.testclient import TestClient

    app = create_app()
    settings = get_settings()

    with TestClient(app, cookies={"hth_session": settings.session_token}) as client:
        with client.websocket_connect("/api/v1/events/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert "session_id" in msg


@pytest.mark.asyncio
async def test_dev_ticket_endpoint_lifecycle() -> None:
    """Verifies that POST /api/v1/auth/dev-ticket succeeds ONLY from Vite dev origin (127.0.0.1:5173)."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        # 1. Valid origin http://127.0.0.1:5173 succeeds
        resp = await client.post(
            "/api/v1/auth/dev-ticket",
            headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:5173"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["token"] == get_settings().session_token
        assert "hth_session=" in resp.headers.get("set-cookie", "")

        # 2. Other origins (even Tauri or external) are forbidden from /dev-ticket
        for forbidden_origin in ["http://127.0.0.1:8765", "http://tauri.localhost"]:
            bad_resp = await client.post(
                "/api/v1/auth/dev-ticket",
                headers={"Host": "127.0.0.1:8765", "Origin": forbidden_origin},
            )
            assert bad_resp.status_code == 403
