"""Async SQLAlchemy engine, session factory, and ORM base with backup/rollback support."""

import datetime
import glob
import os
import shutil
import sqlite3
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from hth.core.settings import get_settings


class Base(DeclarativeBase):
    """Single shared declarative base for all ORM models."""

    pass


def _build_engine() -> AsyncEngine:
    settings = get_settings()
    db_path = settings.db_path
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(url, echo=False)

    # Enable WAL mode on every new raw DBAPI connection (the only safe place).
    # This fires before the first write, including Alembic migrations.
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn: Any, connection_record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with get_session_factory()() as session:
        yield session


async def create_all_tables() -> None:
    """Create all ORM-registered tables.

    IMPORTANT: Import hth.repositories.models BEFORE calling this so that
    all ORM classes are registered on Base.metadata. lifespan.py handles that.
    """
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def create_pre_migration_backup(backup_dir: str | None = None) -> str | None:
    """
    Checkpoint WAL and create an atomic timestamped snapshot before migrations.
    Retains up to 5 most recent backups.
    """
    settings = get_settings()
    db_path = settings.db_path
    if not os.path.exists(db_path):
        return None

    if backup_dir is None:
        backup_dir = os.path.join(os.path.dirname(db_path), "backups")
    os.makedirs(backup_dir, exist_ok=True)

    # Flush WAL via synchronous sqlite3 connection
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        cursor.close()
        conn.close()
    except Exception:
        pass

    # Copy primary db file
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"hth_backup_{timestamp}.db")
    shutil.copy2(db_path, backup_file)

    # Prune old backups (keep latest 5)
    existing_backups = sorted(glob.glob(os.path.join(backup_dir, "hth_backup_*.db")))
    if len(existing_backups) > 5:
        for old in existing_backups[:-5]:
            try:
                os.remove(old)
            except OSError:
                pass

    return backup_file


def verify_database_integrity() -> bool:
    """Run PRAGMA integrity_check on the SQLite database."""
    settings = get_settings()
    db_path = settings.db_path
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return bool(result and result[0] == "ok")
    except Exception:
        return False


def restore_database_from_backup(backup_path: str) -> bool:
    """Restore database from a specified backup file and verify integrity."""
    if not os.path.exists(backup_path):
        return False
    settings = get_settings()
    db_path = settings.db_path
    db_dir = os.path.dirname(db_path)
    os.makedirs(db_dir, exist_ok=True)

    # Clean up any lingering WAL/SHM sidecars
    for ext in ("-wal", "-shm"):
        sidecar = db_path + ext
        if os.path.exists(sidecar):
            try:
                os.remove(sidecar)
            except OSError:
                pass

    shutil.copy2(backup_path, db_path)
    return verify_database_integrity()
