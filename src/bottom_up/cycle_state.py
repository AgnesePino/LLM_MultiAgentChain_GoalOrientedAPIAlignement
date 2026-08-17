"""
Pure state-inspection helpers for the bottom-up HLG-regeneration feedback loop.

A cycle state is only the technical snapshot of one iteration:
- current High-Level Goals;
- current Low-Level Goals;
- branch decisions returned by the Global Goal Evaluator.

Previous states are never ranked and never selected as alternative outputs.
They are kept only by the orchestrator to detect exact/semantic repetition and
to preserve the experimental trace. The cycle evolves one current HLG/LLG
collection and returns that collection when execution stops.
"""

import hashlib
import json

from src.data_model import (
    GlobalGoalCycleIteration,
    GlobalGoalCycleResult,
    GlobalGoalCycleStopReason,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationResult,
    HighLevelGoal,
    HighLevelGoals,
    LowLevelGoals,
)


def _build_structural_decision_state(
    evaluation: GlobalGoalEvaluationResult,
) -> dict[str, object]:
    """Return the rationale-free structural projection of one branch decision."""
    return {
        "decision": evaluation.decision,
        "requires_high_level_regeneration": (
            evaluation.requires_high_level_regeneration
        ),
        "requires_low_level_regeneration": (
            evaluation.requires_low_level_regeneration
        ),
    }


def _build_structural_decisions(
    decisions: dict[str, GlobalGoalEvaluationResult],
) -> dict[str, dict[str, object]]:
    """Build deterministic structural decisions for repeated-state checks."""
    return {
        branch_id: _build_structural_decision_state(evaluation)
        for branch_id, evaluation in sorted(decisions.items())
    }


def _build_state_signature(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
    decisions: dict[str, GlobalGoalEvaluationResult] | None = None,
) -> tuple[str, str]:
    """Build the exact SHA-256 signature of the current cycle snapshot."""
    serialized_state = {
        "high_level_goals": high_level_goals.model_dump(mode="json"),
        "low_level_goals": low_level_goals.model_dump(mode="json"),
        "decisions": _build_structural_decisions(decisions or {}),
    }

    canonical_json = json.dumps(
        serialized_state,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return digest, canonical_json


def _all_expected_branches_confirmed(
    branch_map: dict[str, HighLevelGoal],
    evaluations: dict[str, GlobalGoalEvaluationResult],
    reconstruction_errors: dict[str, str],
    evaluation_errors: dict[str, str],
    empty_branches: list[str],
) -> bool:
    """Return True only when every current branch returned CONFIRM_BRANCH."""
    if reconstruction_errors or evaluation_errors or empty_branches:
        return False

    if set(branch_map) != set(evaluations):
        return False

    return all(
        evaluation.decision == GlobalGoalEvaluationDecision.CONFIRM_BRANCH
        for evaluation in evaluations.values()
    )


def _build_result(
    *,
    converged: bool,
    stop_reason: GlobalGoalCycleStopReason,
    completed_iterations: int,
    max_iterations: int,
    current_high_level_goals: HighLevelGoals,
    current_low_level_goals: LowLevelGoals,
    iterations: list[GlobalGoalCycleIteration],
    added_high_level_goals: list[HighLevelGoal],
    bottom_up_errors: dict[str, str] | None = None,
    evaluation_errors: dict[str, str] | None = None,
    empty_branches: list[str] | None = None,
    coverage_error: str | None = None,
    high_level_regeneration_error: str | None = None,
) -> GlobalGoalCycleResult:
    """Assemble the result from the single current HLG/LLG collection.

    When ``converged`` is True, ``final_*`` is the goal model that passed both
    branch round-trip verification and global documentation coverage and is
    ready for downstream API mapping. Otherwise ``final_*`` is only the latest
    diagnostic snapshot reached before the bounded stop condition.

    ``best_validated_*`` remains ``None`` only because those fields still exist
    in the shared data-model schema for backward compatibility; no best-state
    policy is used by this cycle.
    """
    return GlobalGoalCycleResult(
        converged=converged,
        stop_reason=stop_reason,
        completed_iterations=completed_iterations,
        max_iterations=max_iterations,
        final_high_level_goals=current_high_level_goals,
        final_low_level_goals=current_low_level_goals,
        best_validated_high_level_goals=None,
        best_validated_low_level_goals=None,
        last_candidate_high_level_goals=current_high_level_goals,
        last_candidate_low_level_goals=current_low_level_goals,
        added_high_level_goals=added_high_level_goals,
        iterations=iterations,
        unresolved_bottom_up_errors=bottom_up_errors or {},
        unresolved_global_evaluation_errors=evaluation_errors or {},
        unresolved_empty_branches=empty_branches or [],
        unresolved_documentation_coverage_error=coverage_error,
        unresolved_high_level_regeneration_error=high_level_regeneration_error,
    )
