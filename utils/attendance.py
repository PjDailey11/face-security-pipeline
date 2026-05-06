from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Float, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class AttendanceEvent(Base):
    __tablename__ = "attendance_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    identity: Mapped[str] = mapped_column(String(128))
    similarity: Mapped[float] = mapped_column(Float())
    liveness_prob: Mapped[float] = mapped_column(Float())
    source: Mapped[str | None] = mapped_column(String(256), nullable=True)


class AttendanceLogger:
    """Optional PostgreSQL-backed attendance logging (local DB URL only — privacy-preserving)."""

    def __init__(self, database_url: str) -> None:
        self._engine = create_engine(database_url, pool_pre_ping=True)
        Base.metadata.create_all(self._engine)
        self._session = sessionmaker(bind=self._engine)

    def log(self, identity: str, similarity: float, liveness_prob: float, source: str | None = None) -> None:
        row = AttendanceEvent(
            ts_utc=datetime.now(tz=UTC),
            identity=identity,
            similarity=float(similarity),
            liveness_prob=float(liveness_prob),
            source=source,
        )
        with self._session() as s:
            s.add(row)
            s.commit()
