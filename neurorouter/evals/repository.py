"""SQLite persistence for measured router evaluation reports."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from neurorouter.evals.schemas import EvaluationReport


class EvaluationRepository:
    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
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
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    router_name TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    evaluated_examples INTEGER NOT NULL,
                    report_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_evaluation_runs_completed
                    ON evaluation_runs(completed_at DESC);
                """
            )

    def save(self, report: EvaluationReport) -> EvaluationReport:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id, router_name, dataset_name, started_at, completed_at,
                    evaluated_examples, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(report.run_id),
                    report.router_name,
                    report.dataset_name,
                    report.started_at.isoformat(),
                    report.completed_at.isoformat(),
                    report.metrics.evaluated_examples,
                    report.model_dump_json(),
                ),
            )
        return report

    def get(self, run_id: UUID | str) -> EvaluationReport | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM evaluation_runs WHERE run_id = ?", (str(run_id),)
            ).fetchone()
        return EvaluationReport.model_validate_json(row["report_json"]) if row else None

    def list_reports(self, *, limit: int = 100) -> list[EvaluationReport]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT report_json FROM evaluation_runs ORDER BY completed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [EvaluationReport.model_validate_json(row["report_json"]) for row in rows]
