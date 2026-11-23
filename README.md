# Backend — Python 3.13 + FastAPI Control Service

> **Phase 2+ only.** Placeholder directories are created in Phase 0.
> Do not create source files here until Phase 2 is approved.

---

## Overview

The backend is the local control service for HTH Real Voice Changer. It orchestrates the audio engine child process, manages settings and model metadata, and exposes a typed REST + WebSocket API consumed by the frontend.

## Technology Stack (planned)

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.13.x (64-bit; target 3.13.18, fallback 3.13.17) | Runtime |
| uv | latest | Package + environment manager |
| FastAPI | 0.111+ | HTTP framework |
| Uvicorn | latest | ASGI server |
| Pydantic v2 | 2.x | Data validation + settings |
| pydantic-settings | 2.x | Environment-based configuration |
| SQLAlchemy 2 | 2.x | ORM (async) |
| Alembic | latest | Database migrations |
| structlog | latest | Structured, redacted logging |
| psutil | latest | CPU/RAM/process monitoring |
| pytest | latest | Test runner |
| pytest-asyncio | latest | Async test support |
| httpx | latest | Async HTTP client (for tests) |
| ruff | latest | Linter + formatter |
| mypy | latest | Static type checker |

## Critical Constraints

- **Bind to `127.0.0.1` only** — never `0.0.0.0`
- **No SQL in routers** — routers call services; services call repositories
- **No model weights loaded in this process** — engine runs in isolated child process
- **Per-launch session token** — generated at startup; never committed
- **SQLite WAL** under `%LOCALAPPDATA%\HTH` — never inside the Git repository
- **No `shell=True`** in any subprocess call

## Layer Architecture

```
HTTP Request
     │
     ▼
 API Layer (api/)          ← FastAPI routers; HTTP only; no business logic
     │
     ▼
 Services Layer (services/) ← Orchestration; calls domain + repositories
     │
     ▼
 Domain Layer (domain/)    ← Contracts: VoiceEngine, DeviceInfo, ModelManifest…
     ▲                        No provider-specific types in domain
     │
 Adapters (adapters/)      ← Engine adapters: FakeEngine, Beatrice POC, Applio
 Repositories (repositories/) ← SQLite persistence; no SQL in services
 Audio (audio/)            ← Windows device enumeration, routing checks
```

## Domain Contracts (Phase 2)

```python
# Key domain types (illustrative — implemented in Phase 2)
EngineInfo       # engine id, name, capabilities, status
DeviceInfo       # device id, friendly name, type, sample rates
ModelManifest    # model id, engine_id, format, hash, license_id, consent
LiveConfig       # sample_rate, buffer_size, input_device, output_device
LiveStatus       # engine_state, latency_p50, latency_p95, underruns
AudioMetrics     # frame_count, dropped_frames, cpu_pct, ram_mb
ConsentRecord    # model_id, attested_by, attested_at, license_id
JobState         # job_id, status, progress, error
ErrorEnvelope    # code, message, detail, timestamp
```

## Planned Endpoints (Phase 2 skeleton)

```
GET  /health                       Liveness check
GET  /api/v1/system                System info (OS, CPU, RAM, disk)
GET  /api/v1/devices               Audio device enumeration
GET  /api/v1/engine/capabilities   What the engine supports
GET  /api/v1/engine/status         Current engine state
POST /api/v1/engine/start          Start conversion
POST /api/v1/engine/stop           Stop conversion
POST /api/v1/engine/passthrough    Enable explicit user passthrough
GET  /api/v1/settings              Current settings
PUT  /api/v1/settings              Update settings
GET  /api/v1/models                Model list
POST /api/v1/models/import         Import + validate a model
DELETE /api/v1/models/{id}         Delete a model
POST /api/v1/consent               Record model consent attestation
GET  /api/v1/diagnostics           Latency metrics, process info
WS   /api/v1/events                WebSocket event stream
```

## Development Setup (Phase 2)

```powershell
# Install uv, then install pinned Python 3.13:
uv python install 3.13.18
# (Fallback if standalone 3.13.18 not yet indexed: uv python install 3.13.17)

# Then initialize backend environment:
cd backend
uv venv .venv --python 3.13
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"

# Run
uvicorn hth.core.app:create_app --host 127.0.0.1 --port 8765 --factory

# Check
python -m ruff check .
python -m mypy src/hth/
python -m pytest tests/ -v
```

## Directory Structure

```
backend/
├── src/
│   └── hth/
│       ├── __init__.py
│       ├── api/            FastAPI routers (HTTP layer only)
│       ├── core/           App factory, settings, lifespan, middleware
│       ├── domain/         Protocol contracts — no provider-specific types
│       ├── services/       Business logic orchestration
│       ├── adapters/       Engine adapters (FakeEngine, Beatrice, Applio)
│       ├── audio/          Windows audio device enumeration + routing
│       └── repositories/   SQLite persistence via SQLAlchemy
├── tests/
│   ├── unit/              Isolated unit tests (no I/O)
│   ├── contract/          API contract tests (OpenAPI compliance)
│   └── integration/       Full-stack tests with FakeEngine
├── pyproject.toml         Project metadata + dependencies
├── uv.lock                Locked dependency versions
└── alembic/               Database migration scripts
```
