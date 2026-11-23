"""Authentication router for session validation, bootstrap tickets, and WebSocket ticket issuance.

Security invariant:
- Bound strictly to 127.0.0.1 (Rule 1).
- Single-use ephemeral bootstrap tickets exchanged for in-memory session tokens.
- Issues SameSite=Strict, HttpOnly cookies for web browsers (Tauri & local preview).
- Issues single-use ephemeral WebSocket tickets.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from hth.core.security import ticket_store
from hth.core.settings import get_settings

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class RedeemTicketRequest(BaseModel):
    ticket: str = Field(..., min_length=16, description="Ephemeral single-use bootstrap ticket")


class RedeemTicketResponse(BaseModel):
    ok: bool = True
    token: str = Field(..., description="Active session token for memory storage")


class WsTicketResponse(BaseModel):
    ok: bool = True
    ticket: str = Field(
        ..., description="Short-lived single-use ticket for WebSocket authentication"
    )


class AuthCheckResponse(BaseModel):
    ok: bool = True
    authenticated: bool = True


@router.post(
    "/redeem-ticket",
    response_model=RedeemTicketResponse,
    summary="Redeem one-time bootstrap ticket for session token and cookie",
)
async def redeem_ticket(req: RedeemTicketRequest, response: Response) -> dict[str, Any]:
    """Redeems an ephemeral bootstrap ticket generated at startup.

    Single-use: the ticket is immediately invalidated upon redemption.
    Returns 401 if the ticket is invalid, expired, or already used.
    """
    redeemed = ticket_store.redeem_bootstrap_ticket(req.ticket)
    if not redeemed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired bootstrap ticket",
        )

    settings = get_settings()

    # Issue HttpOnly, SameSite=Strict cookie for session persistence in browser
    # secure=False is intentional for 127.0.0.1 / localhost HTTP loopback
    response.set_cookie(
        key="hth_session",
        value=settings.session_token,
        httponly=True,
        samesite="strict",
        secure=False,
        path="/",
    )

    return {"ok": True, "token": settings.session_token}


@router.post(
    "/dev-ticket",
    response_model=RedeemTicketResponse,
    summary="Ephemeral bootstrap exclusively for local Vite dev loopback",
)
async def dev_bootstrap_ticket(request: Request, response: Response) -> dict[str, Any]:
    """Issues an ephemeral bootstrap session exclusively for the local Vite dev server.

    Exclusively permitted from origin http://127.0.0.1:5173 on local loopback.
    Sets the HttpOnly SameSite=Strict cookie and returns the active session token
    for in-memory closure storage, eliminating any need to publish credentials in URLs.
    """
    origin = request.headers.get("origin", "").strip()
    if origin not in ("http://127.0.0.1:5173", "http://localhost:5173"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Development bootstrap is exclusively permitted from the local Vite dev server (127.0.0.1:5173 or localhost:5173)",
        )

    settings = get_settings()
    response.set_cookie(
        key="hth_session",
        value=settings.session_token,
        httponly=True,
        samesite="strict",
        secure=False,
        path="/",
    )

    return {"ok": True, "token": settings.session_token}


@router.post(
    "/ws-ticket",
    response_model=WsTicketResponse,
    summary="Issue single-use ephemeral WebSocket ticket",
)
async def issue_ws_ticket() -> dict[str, Any]:
    """Issues a short-lived (15s), single-use ticket for authenticating WebSocket connections."""
    ws_ticket = ticket_store.generate_ws_ticket(ttl_seconds=15.0)
    return {"ok": True, "ticket": ws_ticket}


@router.get(
    "/check",
    response_model=AuthCheckResponse,
    summary="Check current session authentication status",
)
async def check_auth() -> dict[str, Any]:
    """Returns 200 if the caller holds a valid session cookie or token."""
    return {"ok": True, "authenticated": True}
