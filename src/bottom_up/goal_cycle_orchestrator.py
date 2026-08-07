"""
Deterministic orchestration of the bounded outer
refinement-abstraction-verification cycle.

The initial top-down pipeline is not executed by this module. The cycle starts
from already-generated HighLevelGoals and LowLevelGoals. During refinement it:
1. reconstructs high-level goal candidates bottom-up;
2. evaluates each branch;
3. asks the original top-down HLG generator to generate only the missing or
   replacement HLGs requested by the evaluator;
4. applies additions and replacements deterministically;
5. preserves the low-level goals of confirmed branches;
6. regenerates only branches that require revision or newly added HLGs;
7. checks global documentation coverage when all branches are confirmed;
8. repeats until all branches are confirmed and coverage is complete.

The HLG generator receives only the request's ``generator_input`` through the
injected callback. It is therefore invoked as a normal top-down generation and
is not exposed to branch identifiers, evaluator decisions, previous attempts,
or replacement metadata.
"""

import hashlib
import json
from pathlib import Path
from typing import Callable

from src.data_model import (
    DocumentationCoverageResult,
    GlobalGoalCycleIteration,
    GlobalGoalCycleResult,
    GlobalGoalEvaluationResult,
    HighLevelGoal,
    HighLevelGoalGenerationRequest,
    HighLevelGoals,
    LowLevelGoal,
    LowLevelGoals,
)
from src.bottom_up.goal_reconstructor import (
    normalize_goal_name,
    reconstruct_all_branches,
)
from src.bottom_up.global_goal_evaluator import (
    evaluate_all_branches,
    evaluate_documentation_coverage,
    save_documentation_coverage,
    save_global_evaluations,
)


# The injected callback receives one validated request and must internally call
# the original top-down HLG generator with request.generator_input and
# feedback=None. The original generator may return one or more HLGs.
HighLevelGoalGenerator = Callable[
    [HighLevelGoalGenerationRequest],
    HighLevelGoals,
]

# The injected callback receives only the HLGs whose LLGs must be generated.
LowLevelGoalRegenerator = Callable[[HighLevelGoals], LowLevelGoals]


def load_global_evaluations(
    evaluation_file: str | Path,
    expected_branch_ids: set[str],
) -> dict[str, GlobalGoalEvaluationResult]:
    """Load one complete evaluator JSON and validate every expected branch."""
    path = Path(evaluation_file)
    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("status") != "READY_FOR_ORCHESTRATION":
        raise ValueError(
            f"Evaluator output '{path}' is not ready for orchestration. "
            f"Status={payload.get('status')!r}, "
            f"missing={payload.get('missing_branch_ids', [])}, "
            f"unexpected={payload.get('unexpected_branch_ids', [])}, "
            f"inconsistent={payload.get('inconsistent_branch_ids', [])}, "
            f"errors={payload.get('errors', {})}."
        )

    errors = payload.get("errors", {})
    if errors:
        raise ValueError(
            f"Evaluator output '{path}' contains unresolved errors: {errors}."
        )

    stored_expected = payload.get("expected_branch_ids")
    if not isinstance(stored_expected, list) or any(
        not isinstance(branch_id, str) for branch_id in stored_expected
    ):
        raise ValueError(
            f"Evaluator output '{path}' has no valid expected_branch_ids list."
        )

    stored_expected_set = set(stored_expected)
    if stored_expected_set != expected_branch_ids:
        raise ValueError(
            f"Evaluator output '{path}' was produced for a different branch set. "
            f"Expected={sorted(expected_branch_ids)}, "
            f"stored={sorted(stored_expected_set)}."
        )

    raw_evaluations = payload.get("evaluations")
    if not isinstance(raw_evaluations, dict):
        raise ValueError(
            f"Evaluator output '{path}' has no valid evaluations object."
        )

    raw_branch_ids = set(raw_evaluations)
    missing = expected_branch_ids - raw_branch_ids
    unexpected = raw_branch_ids - expected_branch_ids
    if missing or unexpected:
        raise ValueError(
            f"Evaluator output '{path}' is not complete. "
            f"Missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )

    evaluations: dict[str, GlobalGoalEvaluationResult] = {}
    for branch_id, raw_evaluation in raw_evaluations.items():
        evaluation = GlobalGoalEvaluationResult.model_validate(raw_evaluation)
        if evaluation.branch_id != branch_id:
            raise ValueError(
                f"Evaluator output '{path}' is inconsistent: dictionary key "
                f"'{branch_id}' does not match evaluation.branch_id "
                f"'{evaluation.branch_id}'."
            )
        evaluations[branch_id] = evaluation

    return evaluations


def load_documentation_coverage(
    coverage_file: str | Path,
) -> DocumentationCoverageResult:
    """Load a persisted documentation-coverage result."""
    path = Path(coverage_file)
    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("status") != "READY_FOR_ORCHESTRATION":
        raise ValueError(
            f"Coverage output '{path}' is not ready for orchestration."
        )

    raw_result = payload.get("documentation_coverage")
    if not isinstance(raw_result, dict):
        raise ValueError(
            f"Coverage output '{path}' has no valid documentation_coverage object."
        )

    return DocumentationCoverageResult.model_validate(raw_result)


def _build_state_signature(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
    decisions: dict[str, GlobalGoalEvaluationResult] | None = None,
) -> str:
    """Build a canonical state hash used to detect non-converging cycles."""
    serialized_state = {
        "high_level_goals": high_level_goals.model_dump(mode="json"),
        "low_level_goals": low_level_goals.model_dump(mode="json"),
        "decisions": {
            branch_id: evaluation.model_dump(mode="json")
            for branch_id, evaluation in sorted((decisions or {}).items())
        },
    }

    canonical_json = json.dumps(
        serialized_state,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _all_expected_branches_confirmed(
    branch_map: dict[str, HighLevelGoal],
    evaluations: dict[str, GlobalGoalEvaluationResult],
    reconstruction_errors: dict[str, str],
    evaluation_errors: dict[str, str],
    empty_branches: list[str],
) -> bool:
    if reconstruction_errors or evaluation_errors or empty_branches:
        return False
    if set(branch_map) != set(evaluations):
        return False
    return all(
        evaluation.decision == "CONFIRM_BRANCH"
        for evaluation in evaluations.values()
    )


def _deduplicate_goals(goals: list[HighLevelGoal]) -> HighLevelGoals:
    result: list[HighLevelGoal] = []
    seen: set[str] = set()

    for goal in goals:
        key = normalize_goal_name(goal.name)
        if key in seen:
            continue
        seen.add(key)
        result.append(goal)

    return HighLevelGoals(goals=result)


def _collect_high_level_generation_requests(
    evaluations: dict[str, GlobalGoalEvaluationResult],
) -> list[HighLevelGoalGenerationRequest]:
    """
    Collect the validated HLG-generation requests emitted by branch evaluation.

    The function also verifies that the orchestration metadata in each request
    is consistent with the evaluator decision and dictionary branch key.
    """
    requests: list[HighLevelGoalGenerationRequest] = []
    seen_request_ids: set[str] = set()

    for branch_id, evaluation in evaluations.items():
        if not evaluation.requires_high_level_regeneration:
            continue

        request = evaluation.generation_request
        if request is None:
            raise ValueError(
                f"Branch '{branch_id}' requires HLG generation but contains "
                "no generation_request."
            )

        if request.origin_branch_id != branch_id:
            raise ValueError(
                f"Request '{request.request_id}' has origin_branch_id "
                f"'{request.origin_branch_id}', expected '{branch_id}'."
            )

        if evaluation.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL":
            if request.action != "REPLACE_EXISTING_HIGH_LEVEL_GOAL":
                raise ValueError(
                    f"Branch '{branch_id}' requires a replacement request, "
                    f"but action is '{request.action}'."
                )
            if request.target_branch_id != branch_id:
                raise ValueError(
                    f"Request '{request.request_id}' targets branch "
                    f"'{request.target_branch_id}', expected '{branch_id}'."
                )

        elif evaluation.decision == "ADD_NEW_HIGH_LEVEL_GOAL":
            if request.action != "ADD_NEW_HIGH_LEVEL_GOAL":
                raise ValueError(
                    f"Branch '{branch_id}' requires an addition request, "
                    f"but action is '{request.action}'."
                )
            if request.target_branch_id is not None:
                raise ValueError(
                    f"Addition request '{request.request_id}' must not target "
                    "an existing branch."
                )

        else:
            raise ValueError(
                f"Branch '{branch_id}' unexpectedly requires HLG generation "
                f"for decision '{evaluation.decision}'."
            )

        if request.request_id in seen_request_ids:
            raise ValueError(
                f"Duplicate HLG generation request id: '{request.request_id}'."
            )

        seen_request_ids.add(request.request_id)
        requests.append(request)

    return requests


def _generate_requested_high_level_goals(
    generate_high_level_goals: HighLevelGoalGenerator,
    requests: list[HighLevelGoalGenerationRequest],
) -> tuple[dict[str, list[HighLevelGoal]], str | None]:
    """
    Execute each request through the original top-down HLG generator.

    The original top-down generator returns ``HighLevelGoals`` and is left
    unchanged. A focused request can therefore still produce one or more HLGs.
    The orchestrator validates every returned goal and keeps the complete list
    associated with the request that produced it.
    """
    generated_by_request: dict[str, list[HighLevelGoal]] = {}

    for request in requests:
        try:
            generated = generate_high_level_goals(request)
        except Exception as exc:
            return {}, (
                f"{type(exc).__name__}: HLG request "
                f"'{request.request_id}' failed: {exc}"
            )

        if not isinstance(generated, HighLevelGoals):
            return {}, (
                "TypeError: generate_high_level_goals must return a "
                "HighLevelGoals instance."
            )

        if not generated.goals:
            return {}, (
                f"ValueError: request '{request.request_id}' returned no HLGs."
            )

        allowed_actor_names = {
            normalize_goal_name(actor.name)
            for actor in request.generator_input.actors.actors
        }
        seen_names_in_request: set[str] = set()
        validated_goals: list[HighLevelGoal] = []

        for generated_goal in generated.goals:
            if (
                not generated_goal.name.strip()
                or not generated_goal.description.strip()
            ):
                return {}, (
                    f"ValueError: request '{request.request_id}' returned an "
                    "HLG with an empty name or description."
                )

            returned_actor_name = normalize_goal_name(
                generated_goal.actor.name
            )
            if returned_actor_name not in allowed_actor_names:
                return {}, (
                    f"ValueError: request '{request.request_id}' returned "
                    f"actor '{generated_goal.actor.name}', which was not "
                    "supplied to the top-down generator."
                )

            normalized_name = normalize_goal_name(generated_goal.name)
            if normalized_name in seen_names_in_request:
                return {}, (
                    f"ValueError: request '{request.request_id}' returned "
                    f"duplicate HLG name '{generated_goal.name}'."
                )

            seen_names_in_request.add(normalized_name)
            validated_goals.append(generated_goal)

        generated_by_request[request.request_id] = validated_goals

    return generated_by_request, None


def _flatten_generated_high_level_goals(
    generated_by_request: dict[str, list[HighLevelGoal]],
) -> list[HighLevelGoal]:
    """Flatten generated HLG lists while preserving request/output order."""
    return [
        goal
        for goals in generated_by_request.values()
        for goal in goals
    ]


def _apply_branch_high_level_generation(
    existing_high_level_goals: dict[str, HighLevelGoal],
    requests: list[HighLevelGoalGenerationRequest],
    generated_by_request: dict[str, list[HighLevelGoal]],
) -> tuple[HighLevelGoals, list[HighLevelGoal]]:
    """
    Apply generated branch replacements and additions deterministically.

    A replacement request may yield one or more HLGs. In that case the target
    branch is removed and the complete generated list is inserted in its
    position. An addition request appends every generated HLG.
    """
    replacements: dict[str, list[HighLevelGoal]] = {}
    additions: list[HighLevelGoal] = []

    for request in requests:
        generated_goals = generated_by_request.get(request.request_id)
        if not generated_goals:
            raise ValueError(
                f"No generated HLGs exist for request '{request.request_id}'."
            )

        if request.action == "REPLACE_EXISTING_HIGH_LEVEL_GOAL":
            target_branch_id = request.target_branch_id
            if target_branch_id is None:
                raise ValueError(
                    f"Request '{request.request_id}' has no target branch."
                )
            if target_branch_id not in existing_high_level_goals:
                raise ValueError(
                    f"Request '{request.request_id}' references unknown branch "
                    f"'{target_branch_id}'."
                )
            if target_branch_id in replacements:
                raise ValueError(
                    "Multiple HLG replacements target branch "
                    f"'{target_branch_id}'."
                )
            replacements[target_branch_id] = generated_goals

        elif request.action == "ADD_NEW_HIGH_LEVEL_GOAL":
            additions.extend(generated_goals)

        else:
            raise ValueError(
                f"Unsupported HLG generation action '{request.action}'."
            )

    updated: list[HighLevelGoal] = []
    seen_names: set[str] = set()

    def append_unique(goal: HighLevelGoal) -> None:
        normalized_name = normalize_goal_name(goal.name)
        if normalized_name in seen_names:
            raise ValueError(
                f"HLG generation produced duplicate goal name '{goal.name}'."
            )
        seen_names.add(normalized_name)
        updated.append(goal)

    for branch_id, original in existing_high_level_goals.items():
        replacement_goals = replacements.get(branch_id)
        if replacement_goals is None:
            append_unique(original)
        else:
            for replacement_goal in replacement_goals:
                append_unique(replacement_goal)

    for new_goal in additions:
        append_unique(new_goal)

    return HighLevelGoals(goals=updated), additions


def _append_coverage_generated_goals(
    current_high_level_goals: HighLevelGoals,
    requests: list[HighLevelGoalGenerationRequest],
    generated_by_request: dict[str, list[HighLevelGoal]],
) -> tuple[HighLevelGoals, list[HighLevelGoal]]:
    """Append all HLGs generated from documentation-coverage requests."""
    current_goals = list(current_high_level_goals.goals)
    seen_names = {
        normalize_goal_name(goal.name)
        for goal in current_goals
    }
    added: list[HighLevelGoal] = []

    for request in requests:
        if request.source != "DOCUMENTATION_COVERAGE":
            raise ValueError(
                f"Coverage request '{request.request_id}' has invalid source "
                f"'{request.source}'."
            )

        if request.action != "ADD_NEW_HIGH_LEVEL_GOAL":
            raise ValueError(
                "Documentation coverage may only add new HLGs."
            )

        generated_goals = generated_by_request.get(request.request_id)
        if not generated_goals:
            raise ValueError(
                f"No generated HLGs exist for request '{request.request_id}'."
            )

        for generated_goal in generated_goals:
            normalized_name = normalize_goal_name(generated_goal.name)
            if normalized_name in seen_names:
                raise ValueError(
                    f"Coverage-generated HLG '{generated_goal.name}' "
                    "duplicates an existing goal."
                )

            seen_names.add(normalized_name)
            current_goals.append(generated_goal)
            added.append(generated_goal)

    return HighLevelGoals(goals=current_goals), added


def _collect_branch_regeneration_targets(
    branch_map: dict[str, HighLevelGoal],
    evaluations: dict[str, GlobalGoalEvaluationResult],
    generated_by_request: dict[str, list[HighLevelGoal]],
) -> tuple[HighLevelGoals, set[str]]:
    """
    Return the HLGs whose LLGs must be generated and the old parent names whose
    current LLGs must be removed.

    If a rewrite request produces multiple HLGs, every generated replacement is
    decomposed. If an addition request produces multiple HLGs, the original
    branch is decomposed again and every newly generated HLG receives its first
    low-level decomposition.
    """
    if set(branch_map) != set(evaluations):
        missing = set(branch_map) - set(evaluations)
        unexpected = set(evaluations) - set(branch_map)
        raise ValueError(
            "Cannot collect regeneration targets from an incomplete "
            f"evaluation set. Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}."
        )

    targets: list[HighLevelGoal] = []
    replaced_parent_names: set[str] = set()

    for branch_id, original in branch_map.items():
        evaluation = evaluations[branch_id]

        if evaluation.decision == "CONFIRM_BRANCH":
            continue

        replaced_parent_names.add(normalize_goal_name(original.name))

        if evaluation.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL":
            request = evaluation.generation_request
            if request is None:
                raise ValueError(
                    f"Branch '{branch_id}' has no generation request."
                )

            replacements = generated_by_request.get(request.request_id)
            if not replacements:
                raise ValueError(
                    f"Branch '{branch_id}' has no generated replacement HLGs."
                )

            targets.extend(replacements)

        elif evaluation.decision == "ADD_NEW_HIGH_LEVEL_GOAL":
            request = evaluation.generation_request
            if request is None:
                raise ValueError(
                    f"Branch '{branch_id}' has no generation request."
                )

            new_goals = generated_by_request.get(request.request_id)
            if not new_goals:
                raise ValueError(
                    f"Branch '{branch_id}' has no generated new HLGs."
                )

            targets.append(original)
            targets.extend(new_goals)

        elif evaluation.decision in {
            "REGENERATE_LOW_LEVEL_GOALS",
            "MATCHES_OTHER_HIGH_LEVEL_GOAL",
        }:
            targets.append(original)

        else:
            raise ValueError(
                f"Branch '{branch_id}' has unsupported decision "
                f"'{evaluation.decision}'."
            )

    return _deduplicate_goals(targets), replaced_parent_names


def _merge_selectively_regenerated_low_level_goals(
    current_low_level_goals: LowLevelGoals,
    regenerated_low_level_goals: LowLevelGoals,
    replaced_parent_names: set[str],
) -> LowLevelGoals:
    """Preserve confirmed branches and replace only requested branches."""
    preserved: list[LowLevelGoal] = [
        goal
        for goal in current_low_level_goals.low_level_goals
        if normalize_goal_name(goal.high_level_associated.name)
        not in replaced_parent_names
    ]

    return LowLevelGoals(
        low_level_goals=[
            *preserved,
            *regenerated_low_level_goals.low_level_goals,
        ]
    )


def _regenerate_selected_branches(
    regenerate_low_level_goals: LowLevelGoalRegenerator,
    targets: HighLevelGoals,
    current_low_level_goals: LowLevelGoals,
    replaced_parent_names: set[str],
) -> tuple[LowLevelGoals | None, str | None]:
    if not targets.goals:
        return None, "ValueError: no high-level goals selected for regeneration."

    try:
        regenerated = regenerate_low_level_goals(targets)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if not isinstance(regenerated, LowLevelGoals):
        return None, (
            "TypeError: regenerate_low_level_goals must return "
            "a LowLevelGoals instance."
        )

    expected_parent_names = {
        normalize_goal_name(goal.name)
        for goal in targets.goals
    }
    returned_parent_names = {
        normalize_goal_name(goal.high_level_associated.name)
        for goal in regenerated.low_level_goals
    }

    missing = expected_parent_names - returned_parent_names
    unexpected = returned_parent_names - expected_parent_names
    if missing or unexpected:
        return None, (
            "ValueError: selective regeneration returned an inconsistent set "
            f"of parent goals. Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}."
        )

    merged = _merge_selectively_regenerated_low_level_goals(
        current_low_level_goals=current_low_level_goals,
        regenerated_low_level_goals=regenerated,
        replaced_parent_names=replaced_parent_names,
    )
    return merged, None


def _build_result(
    *,
    converged: bool,
    stop_reason: str,
    completed_iterations: int,
    max_iterations: int,
    final_high_level_goals: HighLevelGoals,
    final_low_level_goals: LowLevelGoals,
    iterations: list[GlobalGoalCycleIteration],
    added_high_level_goals: list[HighLevelGoal],
    bottom_up_errors: dict[str, str] | None = None,
    evaluation_errors: dict[str, str] | None = None,
    empty_branches: list[str] | None = None,
    coverage_error: str | None = None,
    high_level_regeneration_error: str | None = None,
) -> GlobalGoalCycleResult:
    return GlobalGoalCycleResult(
        converged=converged,
        stop_reason=stop_reason,
        completed_iterations=completed_iterations,
        max_iterations=max_iterations,
        final_high_level_goals=final_high_level_goals,
        final_low_level_goals=final_low_level_goals,
        added_high_level_goals=added_high_level_goals,
        iterations=iterations,
        unresolved_bottom_up_errors=bottom_up_errors or {},
        unresolved_global_evaluation_errors=evaluation_errors or {},
        unresolved_empty_branches=empty_branches or [],
        unresolved_documentation_coverage_error=coverage_error,
        unresolved_high_level_regeneration_error=(
            high_level_regeneration_error
        ),
    )


def run_global_goal_cycle(
    project_description: str,
    initial_high_level_goals: HighLevelGoals,
    initial_low_level_goals: LowLevelGoals,
    generate_high_level_goals: HighLevelGoalGenerator,
    regenerate_low_level_goals: LowLevelGoalRegenerator,
    evaluation_output_directory: str | Path,
    max_iterations: int = 5,
) -> GlobalGoalCycleResult:
    """
    Execute the added bottom-up/refinement pipeline.

    The initial top-down actors, HLGs, and LLGs are treated as input state.
    Confirmed branches retain their current LLGs. HLG generation is invoked only
    for validated generation requests, and LLG generation is invoked only for
    branches that require a new decomposition.

    Every evaluator result is persisted and loaded back before any decision is
    applied. The JSON is therefore the mandatory validated boundary between the
    evaluator and the orchestrator.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be greater than or equal to 1.")

    output_directory = Path(evaluation_output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    current_high_level_goals = initial_high_level_goals.model_copy(deep=True)
    current_low_level_goals = initial_low_level_goals.model_copy(deep=True)

    iteration_traces: list[GlobalGoalCycleIteration] = []
    seen_state_signatures: set[str] = set()
    all_added_high_level_goals: list[HighLevelGoal] = []

    last_bottom_up_errors: dict[str, str] = {}
    last_evaluation_errors: dict[str, str] = {}
    last_empty_branches: list[str] = []

    for iteration_number in range(1, max_iterations + 1):
        (
            reconstructed_goals,
            branch_map,
            bottom_up_errors,
            empty_branches,
            branch_traceability,
        ) = reconstruct_all_branches(
            high_level_goals=current_high_level_goals,
            low_level_goals=current_low_level_goals,
        )

        evaluations, evaluation_errors = evaluate_all_branches(
            project_description=project_description,
            existing_high_level_goals=branch_map,
            reconstructed_high_level_goals=reconstructed_goals,
            empty_branches=empty_branches,
        )

        evaluation_file = (
            output_directory
            / f"iteration_{iteration_number:03d}_global_evaluations.json"
        )
        save_global_evaluations(
            output_file=evaluation_file,
            expected_high_level_goals=branch_map,
            evaluations=evaluations,
            errors=evaluation_errors,
        )

        try:
            evaluations = load_global_evaluations(
                evaluation_file=evaluation_file,
                expected_branch_ids=set(branch_map),
            )
        except Exception as exc:
            evaluation_errors = {
                **evaluation_errors,
                "evaluation_json": f"{type(exc).__name__}: {exc}",
            }

        all_branches_confirmed = _all_expected_branches_confirmed(
            branch_map=branch_map,
            evaluations=evaluations,
            reconstruction_errors=bottom_up_errors,
            evaluation_errors=evaluation_errors,
            empty_branches=empty_branches,
        )

        state_signature = _build_state_signature(
            high_level_goals=current_high_level_goals,
            low_level_goals=current_low_level_goals,
            decisions=evaluations,
        )

        trace = GlobalGoalCycleIteration(
            iteration=iteration_number,
            input_high_level_goals=current_high_level_goals,
            input_low_level_goals=current_low_level_goals,
            branch_traceability=branch_traceability,
            reconstructed_high_level_goals=reconstructed_goals,
            bottom_up_errors=bottom_up_errors,
            empty_branches=empty_branches,
            global_evaluations=evaluations,
            global_evaluation_errors=evaluation_errors,
            all_branches_confirmed=all_branches_confirmed,
            requires_regeneration=not all_branches_confirmed,
            state_signature=state_signature,
        )
        iteration_traces.append(trace)

        last_bottom_up_errors = bottom_up_errors
        last_evaluation_errors = evaluation_errors
        last_empty_branches = empty_branches

        if state_signature in seen_state_signatures:
            return _build_result(
                converged=False,
                stop_reason="REPEATED_STATE_DETECTED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=current_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                bottom_up_errors=bottom_up_errors,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
            )
        seen_state_signatures.add(state_signature)

        if bottom_up_errors:
            return _build_result(
                converged=False,
                stop_reason="BOTTOM_UP_RECONSTRUCTION_FAILED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=current_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                bottom_up_errors=bottom_up_errors,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
            )

        if evaluation_errors:
            return _build_result(
                converged=False,
                stop_reason="GLOBAL_EVALUATION_FAILED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=current_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
            )

        if all_branches_confirmed:
            try:
                coverage = evaluate_documentation_coverage(
                    project_description=project_description,
                    current_high_level_goals=current_high_level_goals,
                )
                coverage_file = (
                    output_directory
                    / f"iteration_{iteration_number:03d}_documentation_coverage.json"
                )
                save_documentation_coverage(coverage_file, coverage)
                coverage = load_documentation_coverage(coverage_file)
            except Exception as exc:
                coverage_error = f"{type(exc).__name__}: {exc}"
                trace.documentation_coverage_error = coverage_error
                return _build_result(
                    converged=False,
                    stop_reason="DOCUMENTATION_COVERAGE_EVALUATION_FAILED",
                    completed_iterations=iteration_number,
                    max_iterations=max_iterations,
                    final_high_level_goals=current_high_level_goals,
                    final_low_level_goals=current_low_level_goals,
                    iterations=iteration_traces,
                    added_high_level_goals=all_added_high_level_goals,
                    coverage_error=coverage_error,
                )

            trace.documentation_coverage = coverage
            trace.documentation_fully_covered = coverage.status == "COMPLETE"

            if coverage.status == "COMPLETE":
                return _build_result(
                    converged=True,
                    stop_reason=(
                        "ALL_BRANCHES_CONFIRMED_AND_DOCUMENTATION_COVERED"
                    ),
                    completed_iterations=iteration_number,
                    max_iterations=max_iterations,
                    final_high_level_goals=current_high_level_goals,
                    final_low_level_goals=current_low_level_goals,
                    iterations=iteration_traces,
                    added_high_level_goals=all_added_high_level_goals,
                )

            generation_requests = coverage.generation_requests
            trace.high_level_generation_requests = generation_requests

            generated_by_request, high_level_error = (
                _generate_requested_high_level_goals(
                    generate_high_level_goals=generate_high_level_goals,
                    requests=generation_requests,
                )
            )

            if high_level_error is not None:
                trace.high_level_regeneration_error = high_level_error
                return _build_result(
                    converged=False,
                    stop_reason="HIGH_LEVEL_REGENERATION_FAILED",
                    completed_iterations=iteration_number,
                    max_iterations=max_iterations,
                    final_high_level_goals=current_high_level_goals,
                    final_low_level_goals=current_low_level_goals,
                    iterations=iteration_traces,
                    added_high_level_goals=all_added_high_level_goals,
                    high_level_regeneration_error=high_level_error,
                )

            trace.generated_high_level_goals = HighLevelGoals(
                goals=_flatten_generated_high_level_goals(generated_by_request)
            )

            try:
                updated_high_level_goals, added_goals = (
                    _append_coverage_generated_goals(
                        current_high_level_goals=current_high_level_goals,
                        requests=generation_requests,
                        generated_by_request=generated_by_request,
                    )
                )
            except Exception as exc:
                high_level_error = f"{type(exc).__name__}: {exc}"
                trace.high_level_regeneration_error = high_level_error
                return _build_result(
                    converged=False,
                    stop_reason="HIGH_LEVEL_REGENERATION_FAILED",
                    completed_iterations=iteration_number,
                    max_iterations=max_iterations,
                    final_high_level_goals=current_high_level_goals,
                    final_low_level_goals=current_low_level_goals,
                    iterations=iteration_traces,
                    added_high_level_goals=all_added_high_level_goals,
                    high_level_regeneration_error=high_level_error,
                )

            trace.newly_added_high_level_goals = added_goals
            trace.updated_high_level_goals = updated_high_level_goals
            trace.requires_regeneration = True
            all_added_high_level_goals.extend(added_goals)

            if iteration_number == max_iterations:
                return _build_result(
                    converged=False,
                    stop_reason="MAX_ITERATIONS_REACHED",
                    completed_iterations=iteration_number,
                    max_iterations=max_iterations,
                    final_high_level_goals=updated_high_level_goals,
                    final_low_level_goals=current_low_level_goals,
                    iterations=iteration_traces,
                    added_high_level_goals=all_added_high_level_goals,
                )

            targets = HighLevelGoals(goals=added_goals)
            replaced_parent_names = {
                normalize_goal_name(goal.name)
                for goal in added_goals
            }
            merged, regeneration_error = _regenerate_selected_branches(
                regenerate_low_level_goals=regenerate_low_level_goals,
                targets=targets,
                current_low_level_goals=current_low_level_goals,
                replaced_parent_names=replaced_parent_names,
            )

            if regeneration_error is not None or merged is None:
                error = (
                    regeneration_error
                    or "Unknown selective regeneration error."
                )
                trace.global_evaluation_errors["low_level_regeneration"] = error
                return _build_result(
                    converged=False,
                    stop_reason="LOW_LEVEL_REGENERATION_FAILED",
                    completed_iterations=iteration_number,
                    max_iterations=max_iterations,
                    final_high_level_goals=updated_high_level_goals,
                    final_low_level_goals=current_low_level_goals,
                    iterations=iteration_traces,
                    added_high_level_goals=all_added_high_level_goals,
                    evaluation_errors={"low_level_regeneration": error},
                )

            trace.regenerated_low_level_goals = merged
            current_high_level_goals = updated_high_level_goals
            current_low_level_goals = merged
            continue

        if iteration_number == max_iterations:
            return _build_result(
                converged=False,
                stop_reason="MAX_ITERATIONS_REACHED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=current_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                bottom_up_errors=bottom_up_errors,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
            )

        try:
            generation_requests = _collect_high_level_generation_requests(
                evaluations
            )
        except Exception as exc:
            high_level_error = f"{type(exc).__name__}: {exc}"
            trace.high_level_regeneration_error = high_level_error
            return _build_result(
                converged=False,
                stop_reason="HIGH_LEVEL_REGENERATION_FAILED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=current_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
                high_level_regeneration_error=high_level_error,
            )

        trace.high_level_generation_requests = generation_requests

        generated_by_request, high_level_error = (
            _generate_requested_high_level_goals(
                generate_high_level_goals=generate_high_level_goals,
                requests=generation_requests,
            )
        )

        if high_level_error is not None:
            trace.high_level_regeneration_error = high_level_error
            return _build_result(
                converged=False,
                stop_reason="HIGH_LEVEL_REGENERATION_FAILED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=current_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
                high_level_regeneration_error=high_level_error,
            )

        if generated_by_request:
            trace.generated_high_level_goals = HighLevelGoals(
                goals=_flatten_generated_high_level_goals(generated_by_request)
            )

        try:
            updated_high_level_goals, newly_added = (
                _apply_branch_high_level_generation(
                    existing_high_level_goals=branch_map,
                    requests=generation_requests,
                    generated_by_request=generated_by_request,
                )
            )

            targets, replaced_parent_names = (
                _collect_branch_regeneration_targets(
                    branch_map=branch_map,
                    evaluations=evaluations,
                    generated_by_request=generated_by_request,
                )
            )
        except Exception as exc:
            high_level_error = f"{type(exc).__name__}: {exc}"
            trace.high_level_regeneration_error = high_level_error
            return _build_result(
                converged=False,
                stop_reason="HIGH_LEVEL_REGENERATION_FAILED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=current_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
                high_level_regeneration_error=high_level_error,
            )

        trace.updated_high_level_goals = updated_high_level_goals
        trace.newly_added_high_level_goals = newly_added
        all_added_high_level_goals.extend(newly_added)

        merged, regeneration_error = _regenerate_selected_branches(
            regenerate_low_level_goals=regenerate_low_level_goals,
            targets=targets,
            current_low_level_goals=current_low_level_goals,
            replaced_parent_names=replaced_parent_names,
        )

        if regeneration_error is not None or merged is None:
            error = regeneration_error or "Unknown selective regeneration error."
            trace.global_evaluation_errors["low_level_regeneration"] = error
            return _build_result(
                converged=False,
                stop_reason="LOW_LEVEL_REGENERATION_FAILED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=updated_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                evaluation_errors={
                    **evaluation_errors,
                    "low_level_regeneration": error,
                },
                empty_branches=empty_branches,
            )

        trace.regenerated_low_level_goals = merged
        current_high_level_goals = updated_high_level_goals
        current_low_level_goals = merged

    return _build_result(
        converged=False,
        stop_reason="MAX_ITERATIONS_REACHED",
        completed_iterations=max_iterations,
        max_iterations=max_iterations,
        final_high_level_goals=current_high_level_goals,
        final_low_level_goals=current_low_level_goals,
        iterations=iteration_traces,
        added_high_level_goals=all_added_high_level_goals,
        bottom_up_errors=last_bottom_up_errors,
        evaluation_errors=last_evaluation_errors,
        empty_branches=last_empty_branches,
    )
