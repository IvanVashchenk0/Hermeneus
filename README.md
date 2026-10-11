# Hermeneus Core

Hermeneus Core is a small backend pipeline that turns text, WAV audio, and optional scene images into validated speech and simulated robot commands. Gemini handles perception and high-level planning; deterministic Python validates and translates that plan. Nothing sends motor commands.

## Setup

Python 3.11 or newer is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` for real model calls. Set `SENTRY_DSN` and `DATABASE_URL` to enable tracing and Tiger Data persistence. Missing optional credentials are reported as unsuccessful integration results in every output; they are never silently treated as success.

Tiger Data must have the TimescaleDB extension enabled. The application creates `pipeline_runs`, `pipeline_events`, and converts the latter to a time-partitioned hypertable on first use.

## Run

First verify the full local pipeline without credentials:

```bash
python main.py --mock --image examples/scene.png --text "Bring me the screwdriver"
python main.py --mock --text "Hello, how are you?"
python main.py --mock --text "Pick up the blue box"
```

Run with Gemini using text or WAV audio:

```bash
python main.py --image examples/scene.png --text "Bring me the screwdriver"
python main.py --image examples/scene.png --audio examples/instruction.wav
```

Add `--tts` to request a local WAV through `pyttsx3`. Platform speech-engine support varies; a TTS failure is returned as a validation error instead of being hidden.

Each run prints JSON and writes the same payload to `outputs/<run_id>.json`. Images/audio are hashed and referenced, never sent to Sentry. Gemini receives media only for the requested inference.

Database queries:

```bash
python main.py --get-run 00000000-0000-0000-0000-000000000000
python main.py --recent
```

Run tests:

```bash
pytest -q
```

## Safety boundary

The model may select only the seven declared high-level skills. The interpreter rejects unknown skills, missing objects, unsupported capabilities, invalid coordinates, blocked endpoints, and paths outside the workspace. Navigation uses 2D grid A* in the world state's named frame and meters. Manipulation remains a high-level command; Hermeneus does not fabricate grasp paths, joint angles, or executable code.

## Structure

- `gemini_engine.py`: structured multimodal perception and planning, plus deterministic mock mode
- `interpreter.py`: validation, translation, A* trajectories, and optional local TTS
- `models.py`: strict Pydantic contracts
- `telemetry.py`: one Sentry transaction per run with bounded, sanitized span data
- `storage.py`: PostgreSQL/TimescaleDB schema, persistence, and queries
- `main.py`: CLI orchestration and JSON output
