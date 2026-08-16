"""
cycle_state.py

State-management helpers for the outer refinement-abstraction-verification
cycle: structural (non-textual) projections of branch decisions used both by
exact and semantic repeated-state detection, the canonical SHA-256 state
signature, best-validated-state scoring, and final-result assembly.

This is a pure structural split out of ``goal_cycle_orchestrator``: every
function below keeps its original logic unchanged.
"""

import hashlib
import json

from src.data_model import HighLevelGoal, HighLevelGoals, LowLevelGoals
from src.bottom_up.goal_reconstructor import normalize_goal_name
from src.bottom_up.models import (
    GlobalGoalCycleIteration,
    GlobalGoalCycleResult,
    GlobalGoalCycleStopReason,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationResult,
)


def _build_structural_decision_state(
    evaluation: GlobalGoalEvaluationResult,
) -> dict[str, object]:
    """
    Structural (non-textual) projection of one branch decision.

    Only the information that defines the decision's operational meaning is
    kept. ``rationale``, ``generation_project_description``, and every other
    LLM free-text field are deliberately excluded, so that two decisions
    that are structurally identical but worded differently by the model
    still collapse to the same exact-state representation.
    """
    structural: dict[str, object] = {
        "decision": evaluation.decision,
        "requires_high_level_regeneration": (
            evaluation.requires_high_level_regeneration
        ),
        "requires_low_level_regeneration": (
            evaluation.requires_low_level_regeneration
        ),
    }

    if (
        evaluation.decision
        == GlobalGoalEvaluationDecision.MATCHES_OTHER_HIGH_LEVEL_GOAL
    ):
        matched = evaluation.matched_high_level_goal
        structural["matched_high_level_goal"] = (
            normalize_goal_name(matched.name) if matched is not None else None
        )

    if evaluation.generation_request is not None:
        structural["generation_request"] = {
            "action": evaluation.generation_request.action,
            "target_branch_id": evaluation.generation_request.target_branch_id,
        }

    return structural


def _build_structural_decisions(
    decisions: dict[str, GlobalGoalEvaluationResult],
) -> dict[str, dict[str, object]]:
    """
    Structural (non-textual) projection of every branch decision, keyed by
    branch_id. Used both by the exact SHA-256 state hash and by semantic
    repeated-state detection, so both criteria ignore the same LLM free text
    (rationale, generation_project_description, ...).
    """
    return {
        branch_id: _build_structural_decision_state(evaluation)
        for branch_id, evaluation in sorted(decisions.items())
    }


def _build_state_signature(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
    decisions: dict[str, GlobalGoalEvaluationResult] | None = None,
) -> tuple[str, str]:
    """
    Build a canonical structural state hash used to detect exactly repeated
    states.

    Returns the SHA-256 digest together with the canonical JSON it was
    derived from, so a caller can additionally verify actual equality of the
    canonical representation on a hash match (defending against the
    theoretical case of a SHA-256 collision).
    """
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
    if reconstruction_errors or evaluation_errors or empty_branches:
        return False
    if set(branch_map) != set(evaluations):
        return False
    return all(
        evaluation.decision == GlobalGoalEvaluationDecision.CONFIRM_BRANCH
        for evaluation in evaluations.values()
    )


def _compute_validated_state_quality(
    branch_map: dict[str, HighLevelGoal],
    evaluations: dict[str, GlobalGoalEvaluationResult],
    empty_branches: list[str],
) -> tuple[float, int, int]:
    """
    Compute a deterministic, unweighted lexicographic quality score for a
    state that has already completed bottom-up reconstruction and global
    branch evaluation with no structural errors (a "validated state").

    A state that only just came out of (re)generation and has not yet gone
    through reconstruction + evaluation in a following iteration is a mere
    candidate state: it must never be scored by this function, and the
    caller is responsible for calling it only once reconstruction/evaluation
    errors have already been ruled out for this state (criterion 1 below).

    The score is ordered lexicographically:
    1. states with reconstruction/evaluation errors are not candidates for
       best_validated_state at all (enforced by the caller, not encoded as a
       score component here);
    2. higher ratio of CONFIRM_BRANCH branches;
    3. fewer empty_branches;
    4. fewer branches that still require regeneration (high-level or
       low-level).

    No arbitrary weights are used: comparison is purely lexicographic via
    tuple ordering. On a full tie, the caller must keep the previously
    recorded best_validated_state by comparing with strict ``>`` rather than
    ``>=``.
    """
    total_branches = len(branch_map)

    confirmed_count = sum(
        evaluation.decision == GlobalGoalEvaluationDecision.CONFIRM_BRANCH
        for evaluation in evaluations.values()
    )

    confirmed_ratio = (
        confirmed_count / total_branches
        if total_branches > 0
        else 0.0
    )

    branches_requiring_regeneration = sum(
        evaluation.requires_low_level_regeneration
        for evaluation in evaluations.values()
    )

    return (
        confirmed_ratio,
        -len(empty_branches),
        -branches_requiring_regeneration,
    )


def _build_result(
    *,
    converged: bool,
    stop_reason: GlobalGoalCycleStopReason,
    completed_iterations: int,
    max_iterations: int,
    current_high_level_goals: HighLevelGoals,
    current_low_level_goals: LowLevelGoals,
    best_high_level_goals: HighLevelGoals | None,
    best_low_level_goals: LowLevelGoals | None,
    iterations: list[GlobalGoalCycleIteration],
    added_high_level_goals: list[HighLevelGoal],
    bottom_up_errors: dict[str, str] | None = None,
    evaluation_errors: dict[str, str] | None = None,
    empty_branches: list[str] | None = None,
    coverage_error: str | None = None,
    high_level_regeneration_error: str | None = None,
) -> GlobalGoalCycleResult:
    """
    Build the final cycle result, applying the final_* policy explicitly:
    - converged: final_* is the converged, validated current state;
    - not converged, a best_validated_state exists: final_* is that state;
    - not converged, no best_validated_state was ever produced (the very
      first iteration already failed structurally): final_* falls back to
      the current/candidate state, since no better information exists.

    best_validated_* and last_candidate_* are always recorded independently
    of this policy, so no experimental information is lost.
    """
    has_validated_state = (
        best_high_level_goals is not None and best_low_level_goals is not None
    )

    if converged:
        final_high_level_goals = current_high_level_goals
        final_low_level_goals = current_low_level_goals
    elif has_validated_state:
        final_high_level_goals = best_high_level_goals
        final_low_level_goals = best_low_level_goals
    else:
        final_high_level_goals = current_high_level_goals
        final_low_level_goals = current_low_level_goals

    return GlobalGoalCycleResult(
        converged=converged,
        stop_reason=stop_reason,
        completed_iterations=completed_iterations,
        max_iterations=max_iterations,
        final_high_level_goals=final_high_level_goals,
        final_low_level_goals=final_low_level_goals,
        best_validated_high_level_goals=best_high_level_goals,
        best_validated_low_level_goals=best_low_level_goals,
        last_candidate_high_level_goals=current_high_level_goals,
        last_candidate_low_level_goals=current_low_level_goals,
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
