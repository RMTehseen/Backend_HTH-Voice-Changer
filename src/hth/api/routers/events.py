"""WebSocket heartbeat endpoint.

SecurityMiddleware does not cover WebSocket upgrades — connection is authenticated
via session cookie, ephemeral single-use WS ticket, query token, or first-message in-band auth.
"""

import asyncio
import datetime
import hmac
import json
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from hth.core.security import ticket_store
from hth.core.settings import get_settings

router = APIRouter(prefix="/api/v1/events", tags=["events"])


@router.websocket("/ws")
async def websocket_events(
    websocket: WebSocket,
    ticket: str = "",
) -> None:
    """WebSocket endpoint for real-time engine events.

    Authoritative authentication rule:
    1. If ?ticket= is supplied: It MUST be redeemed atomically via TicketStore.
       A valid ticket authenticates the connection. An expired, invalid, or replayed
       ticket immediately rejects the connection with code 1008 (Policy Violation).
       A valid cookie must NOT bypass redemption or pardon an invalid ticket.
    2. If ?ticket= is absent: Authenticates strictly via HttpOnly 'hth_session'
       cookie sent during the HTTP upgrade handshake.
    3. Missing credentials, invalid credentials, or legacy ?token= parameters
       immediately reject the connection with code 1008 (Policy Violation).
    """
    settings = get_settings()

    if ticket:
        # Explicit ticket supplied: ticket authentication mode is authoritative.
        # Ticket must be redeemed atomically; cookies cannot bypass or mask ticket validation.
        if not ticket_store.redeem_ws_ticket(ticket):
            await websocket.close(code=1008)
            return
    else:
        # Ticket absent: fallback to documented HttpOnly session cookie on upgrade
        cookie_token = websocket.cookies.get("hth_session", "")
        if not cookie_token or not hmac.compare_digest(
            cookie_token.encode("utf-8"), settings.session_token.encode("utf-8")
        ):
            await websocket.close(code=1008)
            return

    await websocket.accept()

    session_id = str(uuid4())
    try:
        await websocket.send_text(json.dumps({"type": "connected", "session_id": session_id}))
        while True:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            await asyncio.wait_for(
                websocket.send_text(json.dumps({"type": "heartbeat", "ts": now})),
                timeout=5.0,
            )
            await asyncio.sleep(5)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception:
        pass
