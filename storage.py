from __future__ import annotations

import os
from typing import Any

from models import IntegrationResult, PipelineOutput


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    input_metadata JSONB NOT NULL,
    human_instruction TEXT,
    world_state JSONB NOT NULL,
    perception JSONB NOT NULL,
    analysis JSONB NOT NULL,
    interpreter_output JSONB NOT NULL,
    trajectories JSONB NOT NULL,
    speech_text TEXT,
    audio_reference TEXT,
    execution_status TEXT NOT NULL,
    errors JSONB NOT NULL,
    latency_ms JSONB NOT NULL,
    complete_record JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS pipeline_events (
    occurred_at TIMESTAMPTZ NOT NULL,
    run_id UUID NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    payload JSONB NOT NULL
);
SELECT create_hypertable('pipeline_events', 'occurred_at', if_not_exists => TRUE);
"""


def _connect():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    import psycopg

    return psycopg.connect(url)


def initialize_database() -> None:
    with _connect() as connection:
        connection.execute(SCHEMA_SQL)


def persist_run(output: PipelineOutput, events: list[dict[str, Any]]) -> IntegrationResult:
    if not os.getenv("DATABASE_URL"):
        return IntegrationResult(configured=False, succeeded=False, message="DATABASE_URL is not set; run was not persisted")
    try:
        from psycopg.types.json import Jsonb

        success = IntegrationResult(configured=True, succeeded=True, message="Run persisted to Tiger Data")
        # The value is stored inside the same transaction that makes it true.
        output.database_persistence = success
        with _connect() as connection:
            connection.execute(SCHEMA_SQL)
            record = output.model_dump(mode="json")
            interpreter = {
                "commands": record["ordered_robot_actions"],
                "validation": record["interpreter_validation"],
                "tts_file_path": record["tts_file_path"],
            }
            connection.execute(
                """
                INSERT INTO pipeline_runs (
                    run_id, created_at, input_metadata, human_instruction, world_state,
                    perception, analysis, interpreter_output, trajectories, speech_text,
                    audio_reference, execution_status, errors, latency_ms, complete_record
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    output.run_id, output.timestamp, Jsonb(record["input_metadata"]),
                    output.perception.transcribed_instruction, Jsonb(record["world_state"]),
                    Jsonb(record["perception"]), Jsonb(record["analysis"]), Jsonb(interpreter),
                    Jsonb(record["navigation_trajectory"]), output.verbal_response,
                    output.tts_file_path, output.status, Jsonb(output.errors), Jsonb(output.latency_ms),
                    Jsonb(record),
                ),
            )
            for event in events:
                connection.execute(
                    "INSERT INTO pipeline_events (occurred_at, run_id, stage, status, payload) VALUES (%s, %s, %s, %s, %s)",
                    (event["occurred_at"], output.run_id, event["stage"], event["status"], Jsonb(event.get("payload", {}))),
                )
        return success
    except Exception as exc:
        return IntegrationResult(configured=True, succeeded=False, message=f"Database persistence failed: {exc}")


def finalize_run_record(output: PipelineOutput) -> IntegrationResult:
    """Refresh the JSON record after tracing has flushed and final latencies are known."""
    if not output.database_persistence.succeeded:
        return output.database_persistence
    try:
        from psycopg.types.json import Jsonb

        with _connect() as connection:
            connection.execute(
                """UPDATE pipeline_runs
                   SET complete_record = %s, latency_ms = %s, execution_status = %s, errors = %s
                   WHERE run_id = %s""",
                (
                    Jsonb(output.model_dump(mode="json")), Jsonb(output.latency_ms),
                    output.status, Jsonb(output.errors), output.run_id,
                ),
            )
        return output.database_persistence
    except Exception as exc:
        return IntegrationResult(
            configured=True,
            succeeded=False,
            message=f"Core row persisted, but final record update failed: {exc}",
        )


def get_run(run_id: str) -> dict[str, Any] | None:
    from psycopg.rows import dict_row

    with _connect() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            row = cursor.execute(
                "SELECT complete_record FROM pipeline_runs WHERE run_id = %s", (run_id,)
            ).fetchone()
    return row["complete_record"] if row else None


def recent_runs(limit: int = 5) -> list[dict[str, Any]]:
    from psycopg.rows import dict_row

    with _connect() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            rows = cursor.execute(
                """SELECT run_id, created_at, human_instruction, execution_status
                   FROM pipeline_runs ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            ).fetchall()
    return [dict(row) for row in rows]
