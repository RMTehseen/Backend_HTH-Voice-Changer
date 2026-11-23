"""FastAPI application lifespan — startup and shutdown sequences."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from hth.api.dependencies import init_engine_service
from hth.core.database import create_all_tables
from hth.core.logging import configure_logging, get_logger
from hth.core.process_job import SingleInstanceLock, attach_kill_on_close_job
from hth.core.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup order (order matters):
    1. Configure structured logging first so all subsequent startup logs are structured.
    2. Attach to Windows Job Object for process tree auto-cleanup on close.
    3. Acquire single-instance lock to prevent duplicate backends.
    4. Create the DB directory.
    5. Enforce licensing boundaries (prohibit Beatrice in commercial builds).
    6. Import ORM models to register them on Base.metadata.
    7. Create all tables.
    8. Initialise the engine service.
    9. Log startup complete (token masked).
    """
    # 1. Logging
    configure_logging()
    logger = get_logger("hth.lifespan")

    # 2. Windows Job Object auto-cleanup
    attached_job = attach_kill_on_close_job()
    if attached_job:
        logger.info("Windows Job Object attached (KILL_ON_JOB_CLOSE active)")

    # 3. Single-instance lock (skipped in automated test runners)
    settings = get_settings()
    lock: SingleInstanceLock | None = None
    is_testing = bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("TESTING"))
    if settings.single_instance and not is_testing:
        lock_dir = os.path.dirname(settings.db_path)
        os.makedirs(lock_dir, exist_ok=True)
        lock_path = os.path.join(lock_dir, "backend.lock")
        lock = SingleInstanceLock(lock_path)
        if not lock.acquire():
            logger.error(
                "Failed to acquire single-instance lock: another backend is running",
                lock_path=lock_path,
            )
            raise RuntimeError(
                f"Another instance of HTH backend is already running (locked: {lock_path})"
            )
        logger.info("Single-instance lock acquired", lock_path=lock_path)

    # 4. DB directory
    db_dir = os.path.dirname(settings.db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # 5. Licensing boundary check (Rule L-02)
    if settings.commercial_build and settings.engine_adapter.lower() == "beatrice":
        err_msg = (
            "Commercial build violation: Beatrice V2 DSP engine is proprietary/non-commercial "
            "and strictly prohibited from commercial builds without written license."
        )
        logger.critical(err_msg)
        if lock:
            lock.release()
        raise RuntimeError(err_msg)

    # 6. Register ORM models on Base.metadata before create_all_tables
    import hth.repositories.models  # noqa: F401

    # 7. Create tables
    await create_all_tables()

    # 8. Engine service
    init_engine_service(settings.engine_adapter)
    from hth.api.dependencies import init_voice_note_service

    init_voice_note_service()

    # 9. Native background hotkeys (Phase 4)
    from hth.audio.hotkeys import get_hotkey_manager

    get_hotkey_manager().start()

    # 10. Generate ephemeral bootstrap ticket and write to local runtime handshake
    from hth.core.security import ticket_store

    bootstrap_ticket = ticket_store.generate_bootstrap_ticket(ttl_seconds=60.0)
    ticket_file: str | None = None
    try:
        runtime_dir = os.path.dirname(settings.db_path)
        if runtime_dir:
            os.makedirs(runtime_dir, exist_ok=True)
            ticket_file = os.path.join(runtime_dir, "session_ticket.json")
            import json

            with open(ticket_file, "w", encoding="utf-8") as f:
                json.dump({"ticket": bootstrap_ticket, "port": settings.port}, f)
    except Exception as e:
        logger.warning("Could not write runtime session_ticket.json", error=str(e))

    # 11. Log startup (never log the actual token value)
    masked = settings.session_token[:8] + "..." if settings.session_token else "(none)"
    logger.info(
        "HTH backend started",
        host=settings.host,
        port=settings.port,
        engine_adapter=settings.engine_adapter,
        commercial_build=settings.commercial_build,
        db_path=settings.db_path,
        session_token_prefix=masked,
    )

    try:
        yield
    finally:
        logger.info("HTH backend shutting down")
        if ticket_file and os.path.exists(ticket_file):
            try:
                os.remove(ticket_file)
            except Exception:
                pass
        try:
            from hth.audio.hotkeys import get_hotkey_manager

            get_hotkey_manager().stop()
        except Exception:
            pass

        try:
            from hth.audio.recorder import get_test_recorder

            get_test_recorder().purge_all()
        except Exception:
            pass

        if lock:
            lock.release()
            logger.info("Single-instance lock released")
