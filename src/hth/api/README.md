# backend/src/hth/api/ — FastAPI Routers

> **Phase 2+ only.** Placeholder created in Phase 0.

This package contains **only** FastAPI router definitions.

## Rules

- Routers contain HTTP logic only: request parsing, response serialization, error mapping
- No business logic, no SQL, no domain model mutations in this package
- All business operations delegated to `services/`
- No engine-specific imports here

## Planned Routers (Phase 2)

- `health.py` — liveness and readiness endpoints
- `system.py` — OS/hardware info
- `devices.py` — audio device enumeration
- `engine.py` — engine lifecycle (start, stop, passthrough, capabilities, status)
- `settings.py` — app settings CRUD
- `models.py` — model import, list, delete
- `consent.py` — consent attestation
- `diagnostics.py` — latency metrics, process info
- `events.py` — WebSocket event stream
