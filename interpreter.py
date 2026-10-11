from __future__ import annotations

import heapq
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, ContextManager

from models import (
    AnalysisResult,
    InterpreterResult,
    PlannedAction,
    Point2D,
    RobotCommand,
    Skill,
    ValidationResult,
    Waypoint,
    WorldState,
)


class InterpretationError(ValueError):
    pass


def _inside(point: Point2D, world: WorldState) -> bool:
    b = world.workspace
    return b.min_x <= point.x <= b.max_x and b.min_y <= point.y <= b.max_y


def _blocked(point: Point2D, world: WorldState) -> bool:
    return any(
        obstacle.min_x <= point.x <= obstacle.max_x
        and obstacle.min_y <= point.y <= obstacle.max_y
        for obstacle in world.obstacles
    )


def _cell(point: Point2D, world: WorldState) -> tuple[int, int]:
    b = world.workspace
    return (
        round((point.x - b.min_x) / b.resolution),
        round((point.y - b.min_y) / b.resolution),
    )


def _point(cell: tuple[int, int], world: WorldState) -> Point2D:
    b = world.workspace
    return Point2D(
        x=round(b.min_x + cell[0] * b.resolution, 6),
        y=round(b.min_y + cell[1] * b.resolution, 6),
    )


def generate_trajectory(start: Point2D, goal: Point2D, world: WorldState) -> list[Waypoint]:
    """Generate a collision-free 4-neighbor A* path in the declared 2D frame."""
    if not _inside(start, world) or not _inside(goal, world):
        raise InterpretationError("navigation start or target is outside the workspace")
    if _blocked(start, world) or _blocked(goal, world):
        raise InterpretationError("navigation start or target intersects an obstacle")

    start_cell, goal_cell = _cell(start, world), _cell(goal, world)
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start_cell)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
    cost = {start_cell: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal_cell:
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (current[0] + dx, current[1] + dy)
            point = _point(nxt, world)
            if not _inside(point, world) or _blocked(point, world):
                continue
            new_cost = cost[current] + 1
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                priority = new_cost + abs(nxt[0] - goal_cell[0]) + abs(nxt[1] - goal_cell[1])
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal_cell not in came_from:
        raise InterpretationError("no collision-free path exists to the target")

    cells: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = goal_cell
    while cursor is not None:
        cells.append(cursor)
        cursor = came_from[cursor]
    cells.reverse()

    points = [_point(cell, world) for cell in cells]
    points[0] = start
    points[-1] = goal
    elapsed = 0.0
    waypoints: list[Waypoint] = []
    previous = points[0]
    for index, point in enumerate(points):
        if index:
            elapsed += math.hypot(point.x - previous.x, point.y - previous.y) / world.robot.speed_mps
        waypoints.append(Waypoint(x=point.x, y=point.y, t=round(elapsed, 3)))
        previous = point
    return waypoints


def _validate_action(action: PlannedAction, world: WorldState) -> list[str]:
    errors: list[str] = []
    capabilities = set(world.robot.available_skills)
    if action.skill.value not in capabilities:
        errors.append(f"unsupported robot skill: {action.skill.value}")
    target_required = action.skill in {
        Skill.NAVIGATE_TO, Skill.LOCATE_OBJECT, Skill.PICK_UP, Skill.PLACE_OBJECT, Skill.HANDOFF
    }
    known_ids = {obj.id for obj in world.objects}
    if target_required and not action.target_object_id:
        errors.append(f"{action.skill.value} requires a target_object_id")
    elif action.target_object_id and action.target_object_id not in known_ids:
        errors.append(f"unknown target object: {action.target_object_id}")
    if action.skill == Skill.SPEAK and not action.speech_text:
        errors.append("SPEAK requires speech_text")
    return errors


def _generate_tts(text: str, output_path: Path) -> str:
    try:
        import pyttsx3
    except ImportError as exc:
        raise InterpretationError("TTS requested but pyttsx3 is not installed") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()
    if not output_path.exists():
        raise InterpretationError("local TTS engine did not create a WAV file")
    return str(output_path)


def interpret(
    analysis: AnalysisResult,
    world: WorldState,
    *,
    tts_path: Path | None = None,
    stage: Callable[[str], ContextManager[None]] | None = None,
) -> InterpreterResult:
    """Translate a validated model plan without evaluating model-provided code."""
    errors = [error for action in analysis.actions for error in _validate_action(action, world)]
    if analysis.target_object_id and analysis.target_object_id not in {o.id for o in world.objects}:
        errors.append(f"unknown analysis target object: {analysis.target_object_id}")
    if errors:
        return InterpreterResult(validation=ValidationResult(valid=False, errors=errors))

    stage_context = stage or (lambda _name: nullcontext())
    object_map = {obj.id: obj for obj in world.objects}
    current = world.robot.position
    commands: list[RobotCommand] = []
    all_waypoints: list[Waypoint] = []
    for order, action in enumerate(analysis.actions, start=1):
        trajectory: list[Waypoint] = []
        target_position = None
        if action.target_object_id:
            target_position = object_map[action.target_object_id].position
        if action.skill == Skill.NAVIGATE_TO:
            try:
                with stage_context("trajectory_generation"):
                    trajectory = generate_trajectory(current, target_position, world)  # type: ignore[arg-type]
            except InterpretationError as exc:
                errors.append(str(exc))
                break
            current = target_position  # type: ignore[assignment]
            offset = all_waypoints[-1].t if all_waypoints else 0.0
            shifted = [Waypoint(x=w.x, y=w.y, t=round(w.t + offset, 3)) for w in trajectory]
            if all_waypoints and shifted:
                shifted = shifted[1:]
            all_waypoints.extend(shifted)
        commands.append(
            RobotCommand(
                order=order,
                skill=action.skill,
                target_object_id=action.target_object_id,
                target_position=target_position,
                coordinate_frame=world.workspace.frame if target_position else None,
                units=world.workspace.units if target_position else None,
                trajectory=trajectory,
                speech_text=action.speech_text,
            )
        )

    if errors:
        return InterpreterResult(validation=ValidationResult(valid=False, errors=errors))

    audio_path = None
    if tts_path and analysis.speech_text:
        try:
            with stage_context("tts_generation"):
                audio_path = _generate_tts(analysis.speech_text, tts_path)
        except InterpretationError as exc:
            return InterpreterResult(
                commands=commands,
                navigation_trajectory=all_waypoints,
                speech_text=analysis.speech_text,
                validation=ValidationResult(valid=False, errors=[str(exc)]),
            )
    return InterpreterResult(
        commands=commands,
        navigation_trajectory=all_waypoints,
        speech_text=analysis.speech_text,
        tts_file_path=audio_path,
        validation=ValidationResult(valid=True),
    )
