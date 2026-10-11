from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from main import run_pipeline
from models import ResponseType, Skill


@pytest.fixture
def args(tmp_path: Path) -> Namespace:
    return Namespace(
        image=None,
        audio=None,
        text=None,
        world="examples/world.json",
        output_dir=str(tmp_path),
        model=None,
        mock=True,
        tts=False,
    )


def test_speech_only(args: Namespace) -> None:
    args.text = "Hello, how are you?"
    output = run_pipeline(args)
    assert output.response_type == ResponseType.SPEECH
    assert not output.navigation_trajectory
    assert all(command.skill != Skill.NAVIGATE_TO for command in output.ordered_robot_actions)
    assert output.status == "COMPLETED"


def test_robot_action_has_safe_trajectory(args: Namespace) -> None:
    args.text = "Bring me the screwdriver."
    output = run_pipeline(args)
    assert output.response_type == ResponseType.BOTH
    assert output.interpreter_validation.valid
    assert output.navigation_trajectory
    assert [command.order for command in output.ordered_robot_actions] == list(
        range(1, len(output.ordered_robot_actions) + 1)
    )
    world = output.world_state
    for waypoint in output.navigation_trajectory:
        assert world.workspace.min_x <= waypoint.x <= world.workspace.max_x
        assert world.workspace.min_y <= waypoint.y <= world.workspace.max_y
        assert not any(
            obstacle.min_x <= waypoint.x <= obstacle.max_x
            and obstacle.min_y <= waypoint.y <= obstacle.max_y
            for obstacle in world.obstacles
        )


def test_missing_object_requests_clarification(args: Namespace) -> None:
    args.text = "Pick up the blue box."
    output = run_pipeline(args)
    assert output.response_type == ResponseType.CLARIFICATION
    assert output.status == "CLARIFICATION_REQUIRED"
    assert not output.ordered_robot_actions
    assert not output.navigation_trajectory
