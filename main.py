from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from gemini_engine import GeminiEngine
from interpreter import interpret
from models import InputMetadata, IntegrationResult, PipelineOutput, ResponseType, WorldState
from storage import finalize_run_record, get_run, persist_run, recent_runs
from telemetry import Telemetry


def _hash(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path else None


def _media(path: str | None, suffixes: set[str], label: str) -> Path | None:
    if not path:
        return None
    result = Path(path)
    if not result.is_file():
        raise ValueError(f"{label} file does not exist: {result}")
    if result.suffix.lower() not in suffixes:
        raise ValueError(f"{label} must have one of these extensions: {sorted(suffixes)}")
    return result


def run_pipeline(args: argparse.Namespace) -> PipelineOutput:
    run_id = str(uuid4())
    image = _media(args.image, {".jpg", ".jpeg", ".png"}, "image")
    audio = _media(args.audio, {".wav"}, "audio")
    if not args.text and not audio:
        raise ValueError("provide --text or --audio")
    world_path = Path(args.world)
    world = WorldState.model_validate_json(world_path.read_text())
    engine = GeminiEngine(mock=args.mock, model=args.model)
    telemetry = Telemetry(run_id=run_id, model=engine.model)
    events: list[dict] = []

    def event(stage: str, status: str, payload: dict | None = None) -> None:
        events.append({"occurred_at": datetime.now(timezone.utc), "stage": stage, "status": status, "payload": payload or {}})

    with telemetry.trace():
        with telemetry.stage("input_processing", {"has_text": bool(args.text), "has_image": bool(image), "has_audio": bool(audio)}):
            metadata = InputMetadata(
                text_provided=bool(args.text), image_path=str(image) if image else None,
                image_sha256=_hash(image), audio_path=str(audio) if audio else None,
                audio_sha256=_hash(audio), world_path=str(world_path),
            )
            event("input_processing", "ok")
        with telemetry.stage("gemini_perception"):
            perception = engine.perceive(text=args.text, image_path=image, audio_path=audio, world=world)
            telemetry.annotate({
                "goal": perception.intended_goal[:200],
                "matched_object_ids": [item.matched_object_id for item in perception.objects_identified if item.matched_object_id][:20],
                "ambiguity_count": len(perception.ambiguities),
            })
            event("gemini_perception", "ok", {"goal": perception.intended_goal[:200]})
        with telemetry.stage("gemini_analysis"):
            analysis = engine.analyze(perception, world)
            telemetry.annotate({
                "response_type": analysis.response_type.value,
                "status": analysis.status,
                "actions": [action.skill.value for action in analysis.actions][:20],
                "target_object_id": analysis.target_object_id,
            })
            event("gemini_analysis", "ok", {"response_type": analysis.response_type.value, "actions": [a.skill.value for a in analysis.actions]})
        with telemetry.stage("interpreter_translation", {"actions": [a.skill.value for a in analysis.actions]}):
            tts_path = Path(args.output_dir) / f"{run_id}.wav" if args.tts else None
            result = interpret(analysis, world, tts_path=tts_path, stage=lambda name: telemetry.stage(name))
            telemetry.annotate({
                "valid": result.validation.valid,
                "commands": [command.skill.value for command in result.commands][:20],
                "waypoint_count": len(result.navigation_trajectory),
                "errors": result.validation.errors[:5],
            })
            event("interpreter_translation", "ok" if result.validation.valid else "rejected", {"errors": result.validation.errors[:5]})
        if any(command.trajectory for command in result.commands):
            event("trajectory_generation", "ok", {"waypoint_count": len(result.navigation_trajectory)})
        if args.tts:
            event("tts_generation", "ok" if result.tts_file_path else "failed")

        if not result.validation.valid:
            status = "REJECTED"
        elif analysis.response_type == ResponseType.CLARIFICATION:
            status = "CLARIFICATION_REQUIRED"
        else:
            status = "COMPLETED"
        placeholder = IntegrationResult(configured=False, succeeded=False, message="Persistence has not run yet")
        output = PipelineOutput(
            run_id=run_id,
            interpreted_intent=analysis.intent,
            environment_summary=analysis.environment_summary,
            response_type=analysis.response_type,
            verbal_response=result.speech_text,
            tts_file_path=result.tts_file_path,
            ordered_robot_actions=result.commands,
            navigation_trajectory=result.navigation_trajectory,
            interpreter_validation=result.validation,
            status=status,
            errors=result.validation.errors,
            database_persistence=placeholder,
            sentry_tracing=placeholder,
            input_metadata=metadata,
            perception=perception,
            analysis=analysis,
            world_state=world,
            latency_ms=telemetry.latency_ms,
        )
        with telemetry.stage("tiger_data_persistence"):
            output.database_persistence = persist_run(output, events)
            event("tiger_data_persistence", "ok" if output.database_persistence.succeeded else "skipped_or_failed")

    output.latency_ms = telemetry.latency_ms
    output.sentry_tracing = telemetry.finish()
    output.database_persistence = finalize_run_record(output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{run_id}.json"
    destination.write_text(output.model_dump_json(indent=2) + "\n")
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Hermeneus Core robotic interpretation pipeline")
    result.add_argument("--image")
    result.add_argument("--audio")
    result.add_argument("--text")
    result.add_argument("--world", default="examples/world.json")
    result.add_argument("--output-dir", default="outputs")
    result.add_argument("--model")
    result.add_argument("--mock", action="store_true", help="Use deterministic local perception/planning")
    result.add_argument("--tts", action="store_true", help="Generate local WAV speech")
    query = result.add_mutually_exclusive_group()
    query.add_argument("--get-run", metavar="RUN_ID")
    query.add_argument("--recent", action="store_true")
    return result


def main() -> int:
    load_dotenv()
    args = parser().parse_args()
    try:
        if args.get_run:
            print(json.dumps(get_run(args.get_run), indent=2, default=str))
            return 0
        if args.recent:
            print(json.dumps(recent_runs(5), indent=2, default=str))
            return 0
        output = run_pipeline(args)
        print(output.model_dump_json(indent=2))
        return 0 if output.status in {"COMPLETED", "CLARIFICATION_REQUIRED"} else 2
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
