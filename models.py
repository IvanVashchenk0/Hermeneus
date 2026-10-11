from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Point2D(StrictModel):
    x: float
    y: float


class Workspace(StrictModel):
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    resolution: float = Field(gt=0)
    frame: str = "map"
    units: Literal["meters"] = "meters"

    @model_validator(mode="after")
    def valid_bounds(self) -> "Workspace":
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError("workspace minimums must be less than maximums")
        return self


class WorldObject(StrictModel):
    id: str
    label: str
    position: Point2D


class Obstacle(StrictModel):
    id: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float


class RobotState(StrictModel):
    position: Point2D
    available_skills: list[str]
    speed_mps: float = Field(default=0.5, gt=0)


class WorldState(StrictModel):
    coordinate_system: str = "map"
    workspace: Workspace
    robot: RobotState
    objects: list[WorldObject]
    obstacles: list[Obstacle] = Field(default_factory=list)


class VisualObject(StrictModel):
    label: str
    description: str = ""
    matched_object_id: str | None = None


class PerceptionResult(StrictModel):
    transcribed_instruction: str
    intended_goal: str
    objects_identified: list[VisualObject] = Field(default_factory=list)
    environmental_observations: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)


class ResponseType(str, Enum):
    SPEECH = "SPEECH"
    ACTION = "ACTION"
    BOTH = "BOTH"
    CLARIFICATION = "CLARIFICATION"


class Skill(str, Enum):
    NAVIGATE_TO = "NAVIGATE_TO"
    LOCATE_OBJECT = "LOCATE_OBJECT"
    PICK_UP = "PICK_UP"
    PLACE_OBJECT = "PLACE_OBJECT"
    HANDOFF = "HANDOFF"
    STOP = "STOP"
    SPEAK = "SPEAK"


class PlannedAction(StrictModel):
    skill: Skill
    target_object_id: str | None = None
    speech_text: str | None = None


class AnalysisResult(StrictModel):
    intent: str
    environment_summary: str
    response_type: ResponseType
    speech_text: str | None = None
    actions: list[PlannedAction] = Field(default_factory=list)
    target_object_id: str | None = None
    status: Literal["READY", "CLARIFICATION_REQUIRED", "REJECTED"]


class Waypoint(StrictModel):
    x: float
    y: float
    t: float = Field(ge=0, description="Seconds from trajectory start")


class RobotCommand(StrictModel):
    order: int = Field(ge=1)
    skill: Skill
    target_object_id: str | None = None
    target_position: Point2D | None = None
    coordinate_frame: str | None = None
    units: str | None = None
    trajectory: list[Waypoint] = Field(default_factory=list)
    speech_text: str | None = None


class ValidationResult(StrictModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


class InterpreterResult(StrictModel):
    commands: list[RobotCommand] = Field(default_factory=list)
    navigation_trajectory: list[Waypoint] = Field(default_factory=list)
    speech_text: str | None = None
    tts_file_path: str | None = None
    validation: ValidationResult


class IntegrationResult(StrictModel):
    configured: bool
    succeeded: bool
    message: str


class InputMetadata(StrictModel):
    text_provided: bool
    image_path: str | None = None
    image_sha256: str | None = None
    audio_path: str | None = None
    audio_sha256: str | None = None
    world_path: str


class PipelineOutput(StrictModel):
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    interpreted_intent: str
    environment_summary: str
    response_type: ResponseType
    verbal_response: str | None = None
    tts_file_path: str | None = None
    ordered_robot_actions: list[RobotCommand] = Field(default_factory=list)
    navigation_trajectory: list[Waypoint] = Field(default_factory=list)
    interpreter_validation: ValidationResult
    status: Literal["COMPLETED", "CLARIFICATION_REQUIRED", "REJECTED", "FAILED"]
    errors: list[str] = Field(default_factory=list)
    database_persistence: IntegrationResult
    sentry_tracing: IntegrationResult
    input_metadata: InputMetadata
    perception: PerceptionResult
    analysis: AnalysisResult
    world_state: WorldState
    latency_ms: dict[str, float] = Field(default_factory=dict)

    def json_record(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
