"""Task 5 - the database trail.

One row per prediction answers: what was decided, when, by which model, how confidently.
SQLAlchemy Core gives the same code for SQLite (local dev/tests) and PostgreSQL (Compose/cloud);
only DATABASE_URL changes. The uploaded image itself is never stored - only its SHA-256,
so a repeated upload can be recognised without keeping user content.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("filename", String(255), nullable=False),
    Column("image_sha256", String(64), nullable=False, index=True),
    Column("image_width", Integer),
    Column("image_height", Integer),
    Column("label", String(16), nullable=False),          # "ai" | "real"
    Column("confidence", Float, nullable=False),          # prob. of the chosen label
    Column("prob_ai", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model_name", String(128), nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("backend", String(16), nullable=False),        # "local" | "remote"
    Column("latency_ms", Float),
)


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///"):
        from pathlib import Path

        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    # pool_pre_ping: transparently reconnect if Postgres restarted.
    return create_engine(url, pool_pre_ping=True, future=True)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)


def ping(engine: Engine) -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def record(engine: Engine, **row) -> int:
    row.setdefault("created_at", datetime.now(UTC))
    with engine.begin() as c:
        return c.execute(predictions.insert().values(**row)).inserted_primary_key[0]


def history(engine: Engine, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    with engine.connect() as c:
        total = c.execute(select(func.count()).select_from(predictions)).scalar_one()
        rows = c.execute(
            select(predictions).order_by(predictions.c.id.desc()).limit(limit).offset(offset)
        ).mappings().all()
    return [dict(r) for r in rows], total


def stats(engine: Engine) -> dict:
    with engine.connect() as c:
        rows = c.execute(
            select(predictions.c.label, func.count(), func.avg(predictions.c.confidence))
            .group_by(predictions.c.label)
        ).all()
    return {label: {"count": n, "avg_confidence": float(avg or 0)} for label, n, avg in rows}
