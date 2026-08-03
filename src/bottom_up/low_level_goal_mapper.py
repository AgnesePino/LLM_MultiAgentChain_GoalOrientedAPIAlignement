"""
low_level_goal_mapper.py

Deterministic adapter between flat top-down JSON outputs and the bottom-up
pipeline.

Responsibility of this module:
- read the already-generated top-down output;
- keep the existing high-level goals unchanged;
- assign every existing low-level goal to its existing high-level parent;
- create structured LowLevelGoal objects/files required by the bottom-up
  reconstructor;
- never call an LLM;
- never generate, rewrite, evaluate, merge, or delete goals.

The semantic reconstruction of an HLG from its LLGs belongs to
``goal_reconstructor.py``. The comparison and decision belong to
``global_goal_evaluator.py``. Applying those decisions belongs to
``goal_cycle_orchestrator.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LowLevelGoalMappingError(ValueError):
    """Base error raised while preparing low-level goals for bottom-up use."""


class TopDownOutputNotAvailableError(LowLevelGoalMappingError):
    """Raised when the previous top-down phase has produced no readable output."""


class UnsupportedTopDownOutputError(LowLevelGoalMappingError):
    """Raised when no deterministic branch specification exists for a file."""


@dataclass(frozen=True)
class BranchSpecification:
    """
    Python-side mapping metadata for one existing HLG.

    ``actor_index`` identifies the actor already extracted by the top-down phase.
    ``low_level_goal_indices`` contains the indices of the already-generated LLGs
    belonging to that HLG.
    """

    actor_index: int
    low_level_goal_indices: tuple[int, ...]


# Mapping hard-coded, specifico per dataset/modalità di prompting: dice a
# quale attore e a quali indici di LLG corrisponde ogni HLG del top-down.
# Va aggiornato manualmente se cambiano i dataset o il modo in cui il
# top-down genera i suoi output.
# Keys: (dataset_name, prompting_mode, no_llama)
# Values: one BranchSpecification for each HLG, in source order.
BRANCH_SPECIFICATIONS: dict[
    tuple[str, str, bool], tuple[BranchSpecification, ...]
] = {
    # SIA Project 25 26
    ("SIA Project 25 26", "FEW_SHOT", False): (
        BranchSpecification(0, (0, 1)),
        BranchSpecification(1, (5, 6, 7, 8)),
        BranchSpecification(2, (9, 10, 11)),
        BranchSpecification(3, (12, 13)),
        BranchSpecification(4, (14, 15)),
        BranchSpecification(0, (3, 4)),
        BranchSpecification(0, (2,)),
    ),
    ("SIA Project 25 26", "FEW_SHOT", True): (
        BranchSpecification(0, (0, 1, 2)),
        BranchSpecification(1, (3, 4, 5)),
        BranchSpecification(2, (6, 7)),
        BranchSpecification(3, (8, 9)),
        BranchSpecification(4, (10, 11)),
    ),
    ("SIA Project 25 26", "ONE_SHOT", False): (
        BranchSpecification(0, (0, 1, 2, 3, 4, 5)),
        BranchSpecification(1, (6, 7, 8, 9)),
        BranchSpecification(2, (10, 11)),
        BranchSpecification(3, (12, 13)),
        BranchSpecification(4, (14, 15, 16)),
    ),
    ("SIA Project 25 26", "ONE_SHOT", True): (
        BranchSpecification(0, (0, 1, 2)),
        BranchSpecification(1, (3, 4, 5)),
        BranchSpecification(2, (6, 7)),
        BranchSpecification(3, (8, 9)),
        BranchSpecification(4, (10,)),
    ),
    ("SIA Project 25 26", "ZERO_SHOT", False): (
        BranchSpecification(0, (0, 1, 2, 3, 16)),
        BranchSpecification(1, (4, 5, 6, 7)),
        BranchSpecification(2, (8, 9, 17)),
        BranchSpecification(3, (10, 11)),
        BranchSpecification(4, (12, 13, 14)),
        BranchSpecification(5, (15,)),
    ),
    ("SIA Project 25 26", "ZERO_SHOT", True): (
        BranchSpecification(0, (0, 1, 2)),
        BranchSpecification(1, (3, 4, 5, 6)),
        BranchSpecification(2, (7, 8)),
        BranchSpecification(3, (9, 10)),
        BranchSpecification(4, (11, 12)),
    ),

    # London Ambulance Service
    ("London Ambulance Service", "FEW_SHOT", False): (
        BranchSpecification(0, (0, 1)),
        BranchSpecification(0, (2, 8)),
        BranchSpecification(1, (3,)),
        BranchSpecification(2, (4, 9)),
        BranchSpecification(3, (5,)),
        BranchSpecification(4, (6,)),
        BranchSpecification(5, (7,)),
    ),
    ("London Ambulance Service", "FEW_SHOT", True): (
        BranchSpecification(0, (0, 1)),
        BranchSpecification(0, (2,)),
        BranchSpecification(0, (3, 8)),
        BranchSpecification(1, (4,)),
        BranchSpecification(2, (5,)),
        BranchSpecification(0, (6, 9)),
        BranchSpecification(3, (7,)),
    ),
    ("London Ambulance Service", "ONE_SHOT", False): (
        BranchSpecification(0, (0, 1, 2, 3)),
        BranchSpecification(1, (4, 5, 6, 14, 15)),
        BranchSpecification(2, (7, 8)),
        BranchSpecification(3, (9, 10)),
        BranchSpecification(4, (11,)),
        BranchSpecification(5, (12, 13)),
    ),
    ("London Ambulance Service", "ONE_SHOT", True): (
        BranchSpecification(0, (0, 1, 2)),
        BranchSpecification(1, (3, 4, 5)),
        BranchSpecification(2, (6, 7)),
        BranchSpecification(3, (8, 9)),
        BranchSpecification(4, (10, 11)),
    ),
    ("London Ambulance Service", "ZERO_SHOT", False): (
        BranchSpecification(0, (0, 2)),
        BranchSpecification(1, (3, 4)),
        BranchSpecification(2, (5,)),
        BranchSpecification(3, ()),
        BranchSpecification(4, (6,)),
        BranchSpecification(2, (7,)),
        BranchSpecification(2, ()),
        BranchSpecification(0, (1, 8, 9)),
    ),
    ("London Ambulance Service", "ZERO_SHOT", True): (
        BranchSpecification(0, (1, 2)),
        BranchSpecification(1, (3, 4)),
        BranchSpecification(2, (5, 6)),
        BranchSpecification(3, (7, 8)),
        BranchSpecification(4, (0,)),
    ),
}


_REQUIRED_FIELDS = (
    "name",
    "description",
    "actors",
    "highLevelGoals",
    "lowLevelGoals",
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_error_report(
    path: Path,
    *,
    code: str,
    message: str,
    input_directory: Path,
    failures: list[dict[str, str]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "ERROR",
                "error_code": code,
                "message": message,
                "input_directory": str(input_directory),
                "failures": failures or [],
                "created_at_utc": _utc_timestamp(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_source_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LowLevelGoalMappingError(
            f"'{path.name}' is not valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise LowLevelGoalMappingError(
            f"'{path.name}' must contain one JSON object."
        )

    missing = [field for field in _REQUIRED_FIELDS if field not in payload]
    if missing:
        raise LowLevelGoalMappingError(
            f"'{path.name}' is missing required field(s): {missing}."
        )

    for field in ("actors", "highLevelGoals", "lowLevelGoals"):
        values = payload[field]
        if not isinstance(values, list):
            raise LowLevelGoalMappingError(
                f"'{path.name}': '{field}' must be a list."
            )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise LowLevelGoalMappingError(
                f"'{path.name}': '{field}' contains an empty or non-string item."
            )

    if not isinstance(payload["name"], str) or not payload["name"].strip():
        raise LowLevelGoalMappingError(
            f"'{path.name}': 'name' must be a non-empty string."
        )
    if not isinstance(payload["description"], str) or not payload["description"].strip():
        raise LowLevelGoalMappingError(
            f"'{path.name}': 'description' must be a non-empty string."
        )

    return payload


def _infer_variant(path: Path) -> tuple[str, bool]:
    upper_name = path.stem.upper()
    match = re.search(
        r"(?:^|_)(ZERO_SHOT|ONE_SHOT|FEW_SHOT)(?=_|\(|$)",
        upper_name,
    )
    if match is None:
        raise UnsupportedTopDownOutputError(
            f"Cannot infer prompting mode from '{path.name}'."
        )
    return match.group(1), "NOLLAMA" in upper_name.replace("_", "")


def _validate_mapping(
    source_file: Path,
    payload: dict[str, Any],
    specification: tuple[BranchSpecification, ...],
) -> None:
    if len(specification) != len(payload["highLevelGoals"]):
        raise LowLevelGoalMappingError(
            f"'{source_file.name}': mapping has {len(specification)} branches, "
            f"but source has {len(payload['highLevelGoals'])} HLGs."
        )

    # Verifica che ogni indice di attore sia valido e raccoglie tutti gli
    # indici di LLG assegnati a un branch qualsiasi della specifica.
    assigned: list[int] = []
    for hlg_index, branch in enumerate(specification):
        if not 0 <= branch.actor_index < len(payload["actors"]):
            raise LowLevelGoalMappingError(
                f"'{source_file.name}': invalid actor index {branch.actor_index} "
                f"for HLG index {hlg_index}."
            )
        assigned.extend(branch.low_level_goal_indices)

    # Nessun indice di LLG deve essere assegnato a più di un branch.
    duplicate_indices = sorted(
        index for index in set(assigned) if assigned.count(index) > 1
    )
    if duplicate_indices:
        raise LowLevelGoalMappingError(
            f"'{source_file.name}': LLG indices assigned more than once: "
            f"{duplicate_indices}."
        )

    # Ogni LLG del file sorgente deve comparire in esattamente un branch:
    # niente indici mancanti (LLG non assegnati) e niente indici invalidi
    # (fuori dal range dei LLG realmente presenti).
    expected = set(range(len(payload["lowLevelGoals"])))
    actual = set(assigned)
    missing = sorted(expected - actual)
    invalid = sorted(actual - expected)
    if missing or invalid:
        raise LowLevelGoalMappingError(
            f"'{source_file.name}': invalid LLG mapping. "
            f"Missing={missing}, invalid={invalid}."
        )


def _actor_object(name: str, source_index: int) -> dict[str, str]:
    return {
        "name": name,
        "description": (
            "Actor imported unchanged from the saved top-down extraction "
            f"(source actor index {source_index})."
        ),
    }


def map_low_level_goals(
    source_file: str | Path,
    destination_file: str | Path,
) -> dict[str, Any]:
    """
    Group existing LLGs under existing HLGs and write a bottom-up-ready JSON.

    No semantic goal generation or evaluation is performed here.
    """
    source_path = Path(source_file)
    destination_path = Path(destination_file)

    payload = _read_source_json(source_path)
    prompting_mode, no_llama = _infer_variant(source_path)
    dataset_name = payload["name"].strip()

    key = (dataset_name, prompting_mode, no_llama)
    specification = BRANCH_SPECIFICATIONS.get(key)
    if specification is None:
        raise UnsupportedTopDownOutputError(
            "No deterministic LLG-to-HLG mapping is registered for "
            f"dataset={dataset_name!r}, mode={prompting_mode!r}, "
            f"no_llama={no_llama}."
        )

    _validate_mapping(source_path, payload, specification)

    # Costruisce gli oggetti HLG a partire dal testo già estratto dal
    # top-down, associando a ciascuno l'attore indicato dalla specifica.
    high_level_objects: list[dict[str, Any]] = []
    for hlg_index, (hlg_text, branch) in enumerate(
        zip(payload["highLevelGoals"], specification, strict=True)
    ):
        high_level_objects.append(
            {
                "name": f"HLG_{hlg_index + 1:03d}",
                "description": hlg_text,
                "actor": _actor_object(
                    payload["actors"][branch.actor_index],
                    branch.actor_index,
                ),
            }
        )

    low_level_objects: list[dict[str, Any]] = []
    grouped_branches: list[dict[str, Any]] = []

    # Per ogni branch, raggruppa i LLG che gli appartengono (secondo gli
    # indici della specifica) sotto il relativo HLG padre.
    for hlg_index, branch in enumerate(specification):
        parent = high_level_objects[hlg_index]
        branch_llgs: list[dict[str, Any]] = []

        for llg_index in branch.low_level_goal_indices:
            llg = {
                "name": f"LLG_{llg_index + 1:03d}",
                "description": payload["lowLevelGoals"][llg_index],
                "high_level_associated": parent,
            }
            low_level_objects.append(llg)
            branch_llgs.append(llg)

        grouped_branches.append(
            {
                "branch_id": f"branch_{hlg_index + 1:03d}",
                "high_level_goal": parent,
                "low_level_goals": branch_llgs,
                "source_low_level_goal_indices": list(
                    branch.low_level_goal_indices
                ),
                "is_empty_branch": not branch.low_level_goal_indices,
            }
        )

    mapped = {
        "schema_version": "1.0",
        "status": "READY_FOR_BOTTOM_UP_RECONSTRUCTION",
        "source": {
            "file_name": source_path.name,
            "dataset_name": dataset_name,
            "prompting_mode": prompting_mode,
            "no_llama": no_llama,
        },
        "project_description": payload["description"],
        "high_level_goals": {"goals": high_level_objects},
        "low_level_goals": {"low_level_goals": low_level_objects},
        "grouped_branches": grouped_branches,
        "mapping_summary": {
            "high_level_goal_count": len(high_level_objects),
            "low_level_goal_count": len(low_level_objects),
            "empty_branch_ids": [
                branch["branch_id"]
                for branch in grouped_branches
                if branch["is_empty_branch"]
            ],
        },
        "created_at_utc": _utc_timestamp(),
    }

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps(mapped, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return mapped


def create_low_level_mapping_files(
    top_down_output_directory: str | Path,
    mapped_output_directory: str | Path | None = None,
    error_file: str | Path | None = None,
    *,
    fail_on_partial_error: bool = True,
) -> list[Path]:
    """Create one mapped LLG file for every supported top-down JSON."""
    input_dir = Path(top_down_output_directory)
    output_dir = (
        Path(mapped_output_directory)
        if mapped_output_directory is not None
        else input_dir / "bottom_up_inputs"
    )
    error_path = (
        Path(error_file)
        if error_file is not None
        else input_dir.parent / "bottom_up_mapping_error.json"
    )

    if not input_dir.is_dir():
        message = (
            "The top-down output directory does not exist. "
            "The previous top-down phase has not been executed."
        )
        _write_error_report(
            error_path,
            code="TOP_DOWN_OUTPUT_DIRECTORY_NOT_FOUND",
            message=message,
            input_directory=input_dir,
        )
        raise TopDownOutputNotAvailableError(message)

    source_files = sorted(
        path
        for path in input_dir.glob("*.json")
        if not path.name.endswith("_bottom_up_input.json")
        and path.name != error_path.name
    )

    if not source_files:
        message = (
            "The top-down output directory contains no JSON output. "
            "The previous top-down phase has not produced its results."
        )
        _write_error_report(
            error_path,
            code="TOP_DOWN_OUTPUT_DIRECTORY_EMPTY",
            message=message,
            input_directory=input_dir,
        )
        raise TopDownOutputNotAvailableError(message)

    created: list[Path] = []
    failures: list[dict[str, str]] = []

    for source in source_files:
        destination = output_dir / f"{source.stem}_bottom_up_input.json"
        try:
            map_low_level_goals(source, destination)
            created.append(destination)
        except LowLevelGoalMappingError as exc:
            failures.append(
                {
                    "source_file": source.name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    if failures:
        _write_error_report(
            error_path,
            code="LOW_LEVEL_GOAL_MAPPING_FAILED",
            message="One or more top-down outputs could not be mapped.",
            input_directory=input_dir,
            failures=failures,
        )
        if fail_on_partial_error:
            raise LowLevelGoalMappingError(
                "One or more top-down outputs could not be mapped. "
                f"See: {error_path}"
            )
    elif error_path.exists():
        error_path.unlink()

    if not created:
        raise LowLevelGoalMappingError(
            "No bottom-up input file was created. "
            f"See: {error_path}"
        )

    return created


def load_mapped_bottom_up_input(
    mapped_file: str | Path,
) -> tuple[str, Any, Any]:
    """Load mapper output as the objects expected by the reconstructor/cycle."""
    from src.data_model import HighLevelGoals, LowLevelGoals

    path = Path(mapped_file)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "READY_FOR_BOTTOM_UP_RECONSTRUCTION":
        raise LowLevelGoalMappingError(
            f"'{path}' is not ready for bottom-up reconstruction."
        )

    return (
        payload["project_description"],
        HighLevelGoals.model_validate(payload["high_level_goals"]),
        LowLevelGoals.model_validate(payload["low_level_goals"]),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Group already-generated low-level goals under their existing "
            "high-level goals for the bottom-up reconstruction stage."
        )
    )
    parser.add_argument("top_down_output_directory")
    parser.add_argument("--mapped-output-directory", default=None)
    parser.add_argument("--error-file", default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Do not fail the whole run when only some source files fail.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        created = create_low_level_mapping_files(
            top_down_output_directory=args.top_down_output_directory,
            mapped_output_directory=args.mapped_output_directory,
            error_file=args.error_file,
            fail_on_partial_error=not args.allow_partial,
        )
    except LowLevelGoalMappingError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Created {len(created)} mapped bottom-up input file(s):")
    for path in created:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
