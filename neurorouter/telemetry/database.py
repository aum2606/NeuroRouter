"""SQLite persistence for request traces and ordered stage events."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from neurorouter.schemas.trace import StageEvent, TraceRecord
from neurorouter.telemetry.models import TraceUpdate

JSON_COLUMNS = {
    "state",
    "routing_decision",
    "raw_jev_response",
    "execution_plan",
    "quality_gate_result",
    "token_usage",
    "agent_executions",
    "tool_calls",
}

_TRACE_MIGRATIONS = {
    "llm_provider": "TEXT",
    "llm_model": "TEXT",
    "model_tier": "TEXT",
    "token_usage": "TEXT NOT NULL DEFAULT '{}'",
    "agent_executions": "TEXT NOT NULL DEFAULT '[]'",
    "tool_calls": "TEXT NOT NULL DEFAULT '[]'",
}


class TraceRepository:
    """Own SQLite schema setup and durable trace operations."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    request_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    route TEXT,
                    total_latency_ms REAL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    state TEXT,
                    routing_decision TEXT,
                    raw_jev_response TEXT,
                    execution_plan TEXT,
                    synthesis_result TEXT,
                    quality_gate_result TEXT,
                    llm_provider TEXT,
                    llm_model TEXT,
                    model_tier TEXT,
                    token_usage TEXT NOT NULL DEFAULT '{}',
                    agent_executions TEXT NOT NULL DEFAULT '[]',
                    tool_calls TEXT NOT NULL DEFAULT '[]',
                    final_response TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS stage_events (
                    event_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    component TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_ms REAL,
                    dependencies TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    token_usage TEXT NOT NULL,
                    error TEXT,
                    FOREIGN KEY(trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_traces_created_at
                    ON traces(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_stage_events_trace_started
                    ON stage_events(trace_id, started_at);
                """
            )
            existing = {
                row["name"] for row in connection.execute("PRAGMA table_info(traces)").fetchall()
            }
            for column, definition in _TRACE_MIGRATIONS.items():
                if column not in existing:
                    connection.execute(f"ALTER TABLE traces ADD COLUMN {column} {definition}")

    def create_trace(self, trace: TraceRecord) -> TraceRecord:
        values = trace.model_dump(mode="json")
        for column in JSON_COLUMNS:
            if values[column] is not None:
                values[column] = json.dumps(values[column], separators=(",", ":"))
        columns = ", ".join(values)
        placeholders = ", ".join(f":{column}" for column in values)
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO traces ({columns}) VALUES ({placeholders})",  # noqa: S608
                values,
            )
        return trace

    def update_trace(self, trace_id: UUID | str, update: TraceUpdate) -> TraceRecord:
        values = update.model_dump(exclude_none=True, mode="json")
        if not values:
            found = self.get_trace(trace_id)
            if found is None:
                raise KeyError(f"Unknown trace: {trace_id}")
            return found
        values["updated_at"] = datetime.now(UTC).isoformat()
        for column in JSON_COLUMNS & values.keys():
            values[column] = json.dumps(values[column], separators=(",", ":"))
        assignments = ", ".join(f"{column} = :{column}" for column in values)
        values["trace_id"] = str(trace_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE traces SET {assignments} WHERE trace_id = :trace_id",  # noqa: S608
                values,
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown trace: {trace_id}")
        updated = self.get_trace(trace_id)
        if updated is None:  # defensive: update and read use separate connections
            raise KeyError(f"Unknown trace: {trace_id}")
        return updated

    def add_stage_event(self, event: StageEvent) -> StageEvent:
        values = event.model_dump(mode="json")
        for column in ("dependencies", "metadata", "token_usage"):
            values[column] = json.dumps(values[column], separators=(",", ":"))
        columns = ", ".join(values)
        placeholders = ", ".join(f":{column}" for column in values)
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO stage_events ({columns}) VALUES ({placeholders})",  # noqa: S608
                values,
            )
        return event

    def get_trace(self, trace_id: UUID | str) -> TraceRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (str(trace_id),)
            ).fetchone()
        return self._trace_from_row(row) if row else None

    def list_traces(self, *, limit: int = 100, offset: int = 0) -> list[TraceRecord]:
        if limit < 1 or limit > 1000 or offset < 0:
            raise ValueError("limit must be 1..1000 and offset must be non-negative")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM traces ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._trace_from_row(row) for row in rows]

    def list_stage_events(self, trace_id: UUID | str) -> list[StageEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM stage_events WHERE trace_id = ? ORDER BY started_at",
                (str(trace_id),),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    @staticmethod
    def _trace_from_row(row: sqlite3.Row) -> TraceRecord:
        values: dict[str, Any] = dict(row)
        for column in JSON_COLUMNS:
            if values[column] is not None:
                values[column] = json.loads(values[column])
        return TraceRecord.model_validate(values)

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> StageEvent:
        values: dict[str, Any] = dict(row)
        for column in ("dependencies", "metadata", "token_usage"):
            values[column] = json.loads(values[column])
        return StageEvent.model_validate(values)

    def healthcheck(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1
