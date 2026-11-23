"""Per-request security middleware: token auth + Origin/Host validation."""

import hmac
import secrets
import threading
import time
from typing import Awaitable, Callable, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from hth.core.settings import get_settings

# Paths exempt from token authentication (GET only)
EXEMPT_PATHS = {"/health", "/api/v1/health", "/docs", "/openapi.json", "/redoc"}

# Approved origins for local-first loopback and desktop packaging (exact allowlist)
ALLOWED_ORIGINS: frozenset[str] = frozenset(
    {
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8765",
        "http://tauri.localhost",
    }
)


class TicketStore:
    """Thread-safe in-memory store for short-lived, single-use bootstrap and WebSocket tickets.

    Protected by threading.Lock to ensure atomic check-and-invalidate semantics across
    all asynchronous request workers on the single Uvicorn server process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bootstrap_ticket: Optional[str] = None
        self._bootstrap_expiry: float = 0.0
        self._ws_tickets: Dict[str, float] = {}

    def generate_bootstrap_ticket(self, ttl_seconds: float = 60.0) -> str:
        ticket = secrets.token_urlsafe(32)
        with self._lock:
            self._bootstrap_ticket = ticket
            self._bootstrap_expiry = time.monotonic() + ttl_seconds
        return ticket

    def redeem_bootstrap_ticket(self, ticket: str) -> bool:
        if not ticket:
            return False
        with self._lock:
            stored = self._bootstrap_ticket
            expiry = self._bootstrap_expiry
            # Invalidate immediately upon first retrieval attempt (atomic single-use)
            self._bootstrap_ticket = None
            self._bootstrap_expiry = 0.0

            if not stored:
                return False
            if time.monotonic() > expiry:
                return False
            # Constant-time comparison
            return hmac.compare_digest(ticket.encode("utf-8"), stored.encode("utf-8"))

    def generate_ws_ticket(self, ttl_seconds: float = 15.0) -> str:
        now = time.monotonic()
        ticket = secrets.token_urlsafe(32)
        with self._lock:
            # Clean expired tickets
            self._ws_tickets = {t: exp for t, exp in self._ws_tickets.items() if exp > now}
            self._ws_tickets[ticket] = now + ttl_seconds
        return ticket

    def redeem_ws_ticket(self, ticket: str) -> bool:
        if not ticket:
            return False
        now = time.monotonic()
        with self._lock:
            expiry = self._ws_tickets.pop(ticket, None)
            if expiry is None:
                return False
            return now <= expiry


ticket_store = TicketStore()


def verify_session_token(token: str) -> bool:
    """Constant-time verification of session token (Rule S-02)."""
    if not token:
        return False
    expected = get_settings().session_token
    if not expected:
        return False
    return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Validates every non-exempt request against:
    1. Host header — strictly 127.0.0.1 or localhost (DNS rebinding prevention, Rule S-03)
    2. Origin header — loopback HTTP or Tauri custom protocol (Rule S-03)
    3. Session credentials — constant-time compare against hth_session cookie,
       X-HTH-Token, or Bearer Authorization (Rule S-02).

    Bootstrap ticket redemption at POST /api/v1/auth/redeem-ticket is exempt from
    session token checks but still enforces Host and Origin validation.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Allow CORS preflight requests through to CORSMiddleware
        if request.method == "OPTIONS":
            return await call_next(request)

        # --- Host validation (DNS Rebinding prevention) ---
        host = request.headers.get("host", "").strip()
        if not host:
            return JSONResponse(
                {"code": "FORBIDDEN", "message": "Host header required"},
                status_code=403,
            )

        host_name = host.split(":")[0]
        if host_name not in {"127.0.0.1", "localhost"}:
            return JSONResponse(
                {"code": "FORBIDDEN", "message": "Host not allowed"},
                status_code=403,
            )

        # --- Origin validation ---
        origin = request.headers.get("origin", "").strip()
        if origin and origin not in ALLOWED_ORIGINS:
            return JSONResponse(
                {"code": "FORBIDDEN", "message": "Origin not allowed"},
                status_code=403,
            )

        path = request.url.path

        # Exempt certain GET paths from auth entirely
        if request.method == "GET" and path in EXEMPT_PATHS:
            return await call_next(request)

        # Allow unauthenticated ticket redemption (Host/Origin already verified)
        if request.method == "POST" and path in {
            "/api/v1/auth/redeem-ticket",
            "/api/v1/auth/dev-ticket",
        }:
            return await call_next(request)

        # --- Session token check (constant-time) ---
        # 1. Cookie check (HttpOnly, SameSite=Strict)
        provided_token = request.cookies.get("hth_session", "")

        # 2. X-HTH-Token header check
        if not provided_token:
            provided_token = request.headers.get("X-HTH-Token", "")

        # 3. Authorization: Bearer <token>
        if not provided_token:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                provided_token = auth_header[len("Bearer ") :].strip()

        if not verify_session_token(provided_token):
            return JSONResponse(
                {"code": "UNAUTHORIZED", "message": "Invalid or missing session token"},
                status_code=401,
            )

        return await call_next(request)
