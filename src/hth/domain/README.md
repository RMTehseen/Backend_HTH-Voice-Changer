# backend/src/hth/domain/ — Domain Contracts

> **Phase 2+ only.** Placeholder created in Phase 0.

This package contains **only** engine-agnostic domain types and protocols.

## Rules

- No provider-specific types (no Beatrice, Applio, or sounddevice imports here)
- All engine adapters must implement the `VoiceEngine` Protocol defined here
- Domain types use Pydantic v2 `BaseModel` for validation
- No I/O, no network calls, no subprocess calls in this package

## Planned Types (Phase 2)

```python
class VoiceEngine(Protocol): ...     # Engine adapter contract
class EngineInfo(BaseModel): ...     # Engine capabilities and status
class DeviceInfo(BaseModel): ...     # Audio device descriptor
class ModelManifest(BaseModel): ... # Model metadata, hashes, license
class LiveConfig(BaseModel): ...    # Stream configuration
class LiveStatus(BaseModel): ...    # Current engine state + metrics
class AudioMetrics(BaseModel): ...  # Frame stats, dropped frames
class ConsentRecord(BaseModel): ... # Model consent attestation
class JobState(BaseModel): ...      # File conversion job state
class ErrorEnvelope(BaseModel): ... # Typed error response
```
