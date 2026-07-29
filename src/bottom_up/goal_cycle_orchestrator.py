"""
goal_cycle_orchestrator.py

Deterministic orchestration of the bounded outer
refinement-abstraction-verification cycle.

The module does not use an LLM to decide which component to execute.
Python controls the workflow:

1. reconstruct high-level goal candidates bottom-up;
2. evaluate and consolidate the reconstructed candidates;
3. update the high-level goal collection;
4. regenerate the low-level goals when required;
5. repeat until all branches are confirmed or a stopping condition occurs.
"""

import hashlib
import json
from typing import Callable

from src.data_model import (
    GlobalGoalCycleIteration,
    GlobalGoalCycleResult,
    HighLevelGoals,
    LowLevelGoals,
)
from src.bottom_up.goal_reconstructor import reconstruct_all_branches
from src.bottom_up.global_goal_evaluator import (
    apply_global_evaluations,
    evaluate_all_branches,
)


LowLevelGoalRegenerator = Callable[[HighLevelGoals], LowLevelGoals]


def _build_state_signature(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
    decisions: dict[str, object] | None = None,
) -> str:
    """
    Builds a deterministic signature of the current cycle state.

    The signature is used to detect repeated states and prevent oscillating
    or non-progressing outer-loop executions.
    """
    serialized_state = {
        "high_level_goals": high_level_goals.model_dump(mode="json"),
        "low_level_goals": low_level_goals.model_dump(mode="json"),
        "decisions": {
            branch_id: evaluation.model_dump(mode="json")
            for branch_id, evaluation in sorted(
                (decisions or {}).items()
            )
        },
    }

    canonical_json = json.dumps(
        serialized_state,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()


def _all_expected_branches_confirmed(
    branch_map: dict[str, object],
    evaluations: dict[str, object],
    reconstruction_errors: dict[str, str],
    evaluation_errors: dict[str, str],
    empty_branches: list[str],
) -> bool:
    """
    Returns True only when every expected branch has been reconstructed,
    evaluated successfully, and classified as CONFIRM_BRANCH.

    Empty branches and branches with errors prevent convergence.
    """
    if reconstruction_errors or evaluation_errors or empty_branches:
        return False

    expected_branch_ids = set(branch_map)
    evaluated_branch_ids = set(evaluations)

    if expected_branch_ids != evaluated_branch_ids:
        return False

    return all(
        evaluation.decision == "CONFIRM_BRANCH"
        for evaluation in evaluations.values()
    )


def run_global_goal_cycle(
    project_description: str,
    initial_high_level_goals: HighLevelGoals,
    initial_low_level_goals: LowLevelGoals,
    regenerate_low_level_goals: LowLevelGoalRegenerator,
    max_iterations: int = 3,
) -> GlobalGoalCycleResult:
    """
    Executes the complete bounded outer feedback loop.

    Parameters
    ----------
    project_description:
        Normalized project description used by the Global Goal Evaluator.

    initial_high_level_goals:
        High-level goal collection produced by the original pipeline.

    initial_low_level_goals:
        Initial top-down decomposition associated with the high-level goals.

    regenerate_low_level_goals:
        Existing low-level generation pipeline injected by the caller.
        It must accept HighLevelGoals and return LowLevelGoals.

    max_iterations:
        Maximum number of complete outer-loop iterations. Must be at least 1.

    Returns
    -------
    GlobalGoalCycleResult
        Final collections, convergence information, stopping reason, and the
        complete trace of all executed iterations.
    """
    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be greater than or equal to 1."
        )

    current_high_level_goals = initial_high_level_goals
    current_low_level_goals = initial_low_level_goals

    iteration_traces: list[GlobalGoalCycleIteration] = []
    seen_state_signatures: set[str] = set()

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
        )

        all_branches_confirmed = _all_expected_branches_confirmed(
            branch_map=branch_map,
            evaluations=evaluations,
            reconstruction_errors=bottom_up_errors,
            evaluation_errors=evaluation_errors,
            empty_branches=empty_branches,
        )

        regeneration_required = not all_branches_confirmed

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
            requires_regeneration=regeneration_required,
            state_signature=state_signature,
        )

        iteration_traces.append(trace)

        last_bottom_up_errors = bottom_up_errors
        last_evaluation_errors = evaluation_errors
        last_empty_branches = empty_branches

        if all_branches_confirmed:
            return GlobalGoalCycleResult(
                converged=True,
                stop_reason="ALL_BRANCHES_CONFIRMED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=current_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
            )

        if state_signature in seen_state_signatures:
            return GlobalGoalCycleResult(
                converged=False,
                stop_reason="REPEATED_STATE_DETECTED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=current_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                unresolved_bottom_up_errors=bottom_up_errors,
                unresolved_global_evaluation_errors=evaluation_errors,
                unresolved_empty_branches=empty_branches,
            )

        seen_state_signatures.add(state_signature)

        if bottom_up_errors:
            return GlobalGoalCycleResult(
                converged=False,
                stop_reason="BOTTOM_UP_RECONSTRUCTION_FAILED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=current_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                unresolved_bottom_up_errors=bottom_up_errors,
                unresolved_global_evaluation_errors=evaluation_errors,
                unresolved_empty_branches=empty_branches,
            )

        if evaluation_errors:
            return GlobalGoalCycleResult(
                converged=False,
                stop_reason="GLOBAL_EVALUATION_FAILED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=current_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                unresolved_bottom_up_errors=bottom_up_errors,
                unresolved_global_evaluation_errors=evaluation_errors,
                unresolved_empty_branches=empty_branches,
            )

        if iteration_number == max_iterations:
            return GlobalGoalCycleResult(
                converged=False,
                stop_reason="MAX_ITERATIONS_REACHED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=current_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                unresolved_bottom_up_errors=bottom_up_errors,
                unresolved_global_evaluation_errors=evaluation_errors,
                unresolved_empty_branches=empty_branches,
            )

        updated_high_level_goals = apply_global_evaluations(
            existing_high_level_goals=branch_map,
            evaluations=evaluations,
        )

        trace.updated_high_level_goals = updated_high_level_goals

        try:
            regenerated_low_level_goals = regenerate_low_level_goals(
                updated_high_level_goals
            )
        except Exception as exc:
            regeneration_error = f"{type(exc).__name__}: {exc}"

            trace.global_evaluation_errors[
                "low_level_regeneration"
            ] = regeneration_error

            return GlobalGoalCycleResult(
                converged=False,
                stop_reason="LOW_LEVEL_REGENERATION_FAILED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=updated_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                unresolved_bottom_up_errors=bottom_up_errors,
                unresolved_global_evaluation_errors={
                    **evaluation_errors,
                    "low_level_regeneration": regeneration_error,
                },
                unresolved_empty_branches=empty_branches,
            )

        if not isinstance(regenerated_low_level_goals, LowLevelGoals):
            regeneration_error = (
                "TypeError: regenerate_low_level_goals must return "
                "a LowLevelGoals instance."
            )

            trace.global_evaluation_errors[
                "low_level_regeneration"
            ] = regeneration_error

            return GlobalGoalCycleResult(
                converged=False,
                stop_reason="LOW_LEVEL_REGENERATION_FAILED",
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                final_high_level_goals=updated_high_level_goals,
                final_low_level_goals=current_low_level_goals,
                iterations=iteration_traces,
                unresolved_bottom_up_errors=bottom_up_errors,
                unresolved_global_evaluation_errors={
                    **evaluation_errors,
                    "low_level_regeneration": regeneration_error,
                },
                unresolved_empty_branches=empty_branches,
            )

        trace.regenerated_low_level_goals = regenerated_low_level_goals

        current_high_level_goals = updated_high_level_goals
        current_low_level_goals = regenerated_low_level_goals

    return GlobalGoalCycleResult(
        converged=False,
        stop_reason="MAX_ITERATIONS_REACHED",
        completed_iterations=max_iterations,
        max_iterations=max_iterations,
        final_high_level_goals=current_high_level_goals,
        final_low_level_goals=current_low_level_goals,
        iterations=iteration_traces,
        unresolved_bottom_up_errors=last_bottom_up_errors,
        unresolved_global_evaluation_errors=last_evaluation_errors,
        unresolved_empty_branches=last_empty_branches,
    )
