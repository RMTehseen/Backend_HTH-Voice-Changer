# backend/src/hth/adapters/ — Engine Adapters

> **Phase 2+ only.** Placeholder created in Phase 0.

This package contains one sub-package per engine adapter, each implementing the `VoiceEngine` protocol defined in `domain/`.

## Rules

- Each adapter lives in its own sub-package: `adapters/<engine_name>/`
- Adapters import engine-specific code only within their own sub-package
- Adapters are **never** imported directly into the FastAPI process for audio/model loading — they run in a child process
- The `FakeEngine` adapter is the only one permitted to run in-process (for tests)

## Planned Adapters (Phase 2+)

| Sub-package | Engine | Phase | Status |
|------------|--------|-------|--------|
| `fake/` | FakeEngine (test stub) | 2 | ⏳ |
| `beatrice/` | Beatrice Windows POC | 3 | 📋 POC-ONLY |
| `applio/` | Applio 3.7.1 (optional) | 3 | ⏳ Optional |

## Beatrice Adapter Note

```python
# PERSONAL_POC_ONLY — do not enable in commercial build
# Proprietary VST engine: personal/non-commercial restrictions
# JVS Corpus models: non-commercial
COMMERCIAL_DISTRIBUTION_ENABLED = False  # never change to True without written rights
```
