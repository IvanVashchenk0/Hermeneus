from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path

from models import (
    AnalysisResult,
    PerceptionResult,
    PlannedAction,
    ResponseType,
    Skill,
    VisualObject,
    WorldState,
)


DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiEngine:
    def __init__(self, *, mock: bool = False, model: str | None = None) -> None:
        self.mock = mock
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.client = None
        if not mock:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is required unless --mock is used")
            from google import genai

            self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _media_part(path: Path):
        from google.genai import types

        mime_type = mimetypes.guess_type(path)[0]
        if not mime_type:
            raise ValueError(f"cannot determine media type for {path}")
        return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)

    def perceive(
        self,
        *,
        text: str | None,
        image_path: Path | None,
        audio_path: Path | None,
        world: WorldState,
    ) -> PerceptionResult:
        if self.mock:
            return self._mock_perception(text or "", world)
        from google.genai import types

        parts = [
            types.Part.from_text(
                text=(
                    "Interpret the supplied human instruction and scene. Identify objects only when supported "
                    "by the media. Never infer coordinates. Match labels to these known IDs when reliable: "
                    + json.dumps([{"id": o.id, "label": o.label} for o in world.objects])
                    + (f"\nPlain-text instruction: {text}" if text else "")
                )
            )
        ]
        if image_path:
            parts.append(self._media_part(image_path))
        if audio_path:
            parts.append(self._media_part(audio_path))
        response = self.client.models.generate_content(  # type: ignore[union-attr]
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PerceptionResult,
                temperature=0,
            ),
        )
        return PerceptionResult.model_validate_json(response.text)

    def analyze(self, perception: PerceptionResult, world: WorldState) -> AnalysisResult:
        if self.mock:
            return self._mock_analysis(perception, world)
        from google.genai import types

        prompt = (
            "Plan a safe response using only available_skills and existing object IDs. "
            "Do not invent targets, coordinates, skills, or capabilities. If uncertain or impossible, use "
            "CLARIFICATION with no physical actions. World state and perception follow.\nWORLD:\n"
            + world.model_dump_json()
            + "\nPERCEPTION:\n"
            + perception.model_dump_json()
        )
        response = self.client.models.generate_content(  # type: ignore[union-attr]
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisResult,
                temperature=0,
            ),
        )
        return AnalysisResult.model_validate_json(response.text)

    @staticmethod
    def _mock_perception(text: str, world: WorldState) -> PerceptionResult:
        lowered = text.lower()
        matches = [
            VisualObject(label=obj.label, matched_object_id=obj.id, description="Known from mock world state")
            for obj in world.objects
            if obj.label.lower() in lowered
        ]
        ambiguity = [] if text.strip() else ["No text instruction was supplied in mock mode"]
        return PerceptionResult(
            transcribed_instruction=text,
            intended_goal=text,
            objects_identified=matches,
            environmental_observations=["Mock perception; no visual claims were generated"],
            ambiguities=ambiguity,
        )

    @staticmethod
    def _mock_analysis(perception: PerceptionResult, world: WorldState) -> AnalysisResult:
        text = perception.transcribed_instruction.lower()
        if "hello" in text or "how are you" in text:
            return AnalysisResult(
                intent="Greet and converse with the human",
                environment_summary="No physical action is needed.",
                response_type=ResponseType.SPEECH,
                speech_text="Hello! I'm operating normally and ready to help.",
                actions=[PlannedAction(skill=Skill.SPEAK, speech_text="Hello! I'm operating normally and ready to help.")],
                status="READY",
            )
        requested = next((o for o in world.objects if o.label.lower() in text), None)
        capabilities = set(world.robot.available_skills)
        if "bring" in text and requested and {"NAVIGATE_TO", "PICK_UP", "HANDOFF"} <= capabilities:
            handoff = next((o for o in world.objects if o.label.lower() == "handoff zone"), None)
            if handoff:
                speech = f"I'll bring you the {requested.label}."
                return AnalysisResult(
                    intent=f"Bring {requested.label} to the human",
                    environment_summary=f"{requested.label} and handoff zone have known world-state positions.",
                    response_type=ResponseType.BOTH,
                    speech_text=speech,
                    target_object_id=requested.id,
                    actions=[
                        PlannedAction(skill=Skill.SPEAK, speech_text=speech),
                        PlannedAction(skill=Skill.NAVIGATE_TO, target_object_id=requested.id),
                        PlannedAction(skill=Skill.PICK_UP, target_object_id=requested.id),
                        PlannedAction(skill=Skill.NAVIGATE_TO, target_object_id=handoff.id),
                        PlannedAction(skill=Skill.HANDOFF, target_object_id=requested.id),
                    ],
                    status="READY",
                )
        missing_label = "blue box" if "blue box" in text else "requested object"
        return AnalysisResult(
            intent=perception.intended_goal or "Unknown",
            environment_summary="The request cannot be grounded safely in the current world state and capabilities.",
            response_type=ResponseType.CLARIFICATION,
            speech_text=f"I cannot safely act: the {missing_label} or a required capability is unavailable.",
            actions=[],
            status="CLARIFICATION_REQUIRED",
        )
