"""Deterministically group each LLG under its top-down HLG parent."""

import json
from pathlib import Path

from src.data_model import GoalBranch, HighLevelGoals, LowLevelGoals


def _key(value: str) -> str:
    return " ".join(value.casefold().split())


def group_low_level_goals(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
) -> list[GoalBranch]:
    branches = [
        GoalBranch(
            branch_id=f"branch_{index:03d}",
            high_level_goal=goal,
            low_level_goals=[],
        )
        for index, goal in enumerate(high_level_goals.goals, start=1)
    ]
    by_name = {_key(branch.high_level_goal.name): branch for branch in branches}

    for llg in low_level_goals.low_level_goals:
        branch = by_name.get(_key(llg.high_level_associated.name))
        if branch is None:
            raise ValueError(
                f"LLG '{llg.name}' refers to unknown HLG "
                f"'{llg.high_level_associated.name}'."
            )
        branch.low_level_goals.append(llg)

    return branches


def map_low_level_goals(source_file: str | Path, destination_file: str | Path) -> dict:
    source = Path(source_file)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "structuredHighLevelGoals" not in payload or "structuredLowLevelGoals" not in payload:
        raise ValueError("The top-down output must contain structured HLG and LLG fields.")

    hlgs = HighLevelGoals.model_validate(payload["structuredHighLevelGoals"])
    llgs = LowLevelGoals.model_validate(payload["structuredLowLevelGoals"])
    branches = group_low_level_goals(hlgs, llgs)
    mapped = {
        "project_description": payload["description"],
        "high_level_goals": hlgs.model_dump(mode="json"),
        "low_level_goals": llgs.model_dump(mode="json"),
        "branches": [branch.model_dump(mode="json") for branch in branches],
    }
    destination = Path(destination_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(mapped, indent=2, ensure_ascii=False), encoding="utf-8")
    return mapped


def load_mapped_bottom_up_input(mapped_file: str | Path):
    payload = json.loads(Path(mapped_file).read_text(encoding="utf-8"))
    return (
        payload["project_description"],
        HighLevelGoals.model_validate(payload["high_level_goals"]),
        LowLevelGoals.model_validate(payload["low_level_goals"]),
    )
