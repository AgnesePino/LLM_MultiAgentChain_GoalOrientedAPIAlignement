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
or replacement metadata. After generation, semantic duplicate clusters may
contain newly generated HLGs and HLGs already present in the cycle state. The
Global Goal Evaluator selects exactly one project-grounded GORE representative;
the update layer removes every losing duplicate deterministically.

State-management helpers (structural decision projections, the canonical
state signature, best-validated-state scoring, and final-result assembly)
live in ``cycle_state.py``. Helpers that apply evaluator decisions to the
current HLG/LLG state (deduplication, HLG-generation requests, selective
low-level regeneration) live in ``goal_update.py``. This module remains
responsible for reconstruct -> evaluate -> persist/load evaluator results ->
detect stop conditions -> invoke update/regeneration helpers -> advance to
the next iteration.
"""

import json
from pathlib import Path

from src.data_model import HighLevelGoal, HighLevelGoals, LowLevelGoals
from src.bottom_up.goal_reconstructor import (
    assign_local_goal_ids,
    group_low_level_goals_by_branch,
    normalize_goal_name,
    reconstruct_all_branches,
)
from src.bottom_up.global_goal_evaluator import (
    evaluate_all_branches,
    evaluate_documentation_coverage,
    save_documentation_coverage,
    save_global_evaluations,
    select_best_duplicate_high_level_goal,
)
from src.bottom_up.models import (
    DocumentationCoverageResult,
    DocumentationCoverageStatus,
    GlobalGoalCycleIteration,
    GlobalGoalCycleResult,
    GlobalGoalCycleStopReason,
    GlobalGoalEvaluationResult,
    LowLevelGoal,
)
from src.bottom_up.cycle_state import (
    _all_expected_branches_confirmed,
    _build_result,
    _build_state_signature,
    _build_structural_decisions,
    _compute_validated_state_quality,
)
from src.bottom_up.goal_update import (
    HighLevelGoalGenerator,
    LowLevelGoalRegenerator,
    _apply_branch_high_level_generation,
    _append_coverage_generated_goals,
    _collect_branch_regeneration_targets,
    _collect_high_level_generation_requests,
    _flatten_generated_high_level_goals,
    _generate_requested_high_level_goals,
    _regenerate_selected_branches,
)
from src.bottom_up.semantic_similarity import states_semantically_equivalent

# Single default for the outer cycle's iteration bound. This is a safety
# bound to guarantee termination, not a scientifically optimal value: callers
# running experiments are expected to override it explicitly when needed.
DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS = 5


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


def _build_branch_low_level_goal_index(
    branch_map: dict[str, HighLevelGoal],
    low_level_goals: LowLevelGoals,
) -> dict[str, dict[str, LowLevelGoal]]:
    """
    Deterministically maps each branch's full low-level-goal ids
    ('branch_001_llg_001', ...) to the corresponding LowLevelGoal, so the
    Global Evaluator can see LLG text and not just ids.

    Rebuilds the exact same branch grouping and per-branch local numbering
    already used internally by reconstruct_all_branches /
    reconstruct_high_level_goal (group_low_level_goals_by_branch +
    assign_local_goal_ids), so the ids line up with
    BottomUpHighLevelGoal.source_low_level_goal_ids /
    supporting_low_level_goal_ids / non_supporting_low_level_goal_ids without
    changing reconstruct_all_branches' public return signature (which other
    callers, e.g. the experiments notebook, already unpack positionally).
    """
    grouped = group_low_level_goals_by_branch(low_level_goals, branch_map)

    return {
        branch_id: {
            f"{branch_id}_{local_id}": goal
            for local_id, goal in assign_local_goal_ids(goals).items()
        }
        for branch_id, goals in grouped.items()
    }


def run_global_goal_cycle(
    project_description: str,
    initial_high_level_goals: HighLevelGoals,
    initial_low_level_goals: LowLevelGoals,
    generate_high_level_goals: HighLevelGoalGenerator,
    regenerate_low_level_goals: LowLevelGoalRegenerator,
    evaluation_output_directory: str | Path,
    max_iterations: int = DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS,
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

    # best_validated_state: only ever set from a state that has completed
    # bottom-up reconstruction + global branch evaluation with no structural
    # errors (see _compute_validated_state_quality). A state produced by a
    # (re)generation step is a mere candidate until it goes through that
    # validation in a following iteration, so it must never be written here
    # directly.
    best_high_level_goals: HighLevelGoals | None = None
    best_low_level_goals: LowLevelGoals | None = None
    best_score: tuple[float, int, int] | None = None

    iteration_traces: list[GlobalGoalCycleIteration] = []
    # Exact repeated-state detection: maps a structural state hash to the
    # canonical JSON it was derived from, so an eventual hash match can be
    # confirmed as an actual equality (defending against the theoretical
    # case of a SHA-256 collision) rather than trusted on the digest alone.
    seen_state_signatures: dict[str, str] = {}
    # Semantic repeated-state detection: the raw HLG/LLG collections of every
    # previously seen state, together with the structural (non-textual)
    # projection of its branch decisions, compared against the current state
    # when no exact repeat was found. Both the HLG/LLG collections (via
    # embeddings + cosine similarity, see semantic_similarity.py) AND the
    # structural decisions (via deterministic equality) must match: a state
    # with semantically equivalent HLG/LLG but a different decision (e.g.
    # REGENERATE_LOW_LEVEL_GOALS vs. CONFIRM_BRANCH) has actually progressed
    # and must not be treated as a stall.
    seen_states: list[
        tuple[HighLevelGoals, LowLevelGoals, dict[str, dict[str, object]]]
    ] = []
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

        branch_low_level_goals = _build_branch_low_level_goal_index(
            branch_map=branch_map,
            low_level_goals=current_low_level_goals,
        )

        evaluations, evaluation_errors = evaluate_all_branches(
            project_description=project_description,
            existing_high_level_goals=branch_map,
            reconstructed_high_level_goals=reconstructed_goals,
            empty_branches=empty_branches,
            branch_low_level_goals=branch_low_level_goals,
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

        state_signature, state_canonical_json = _build_state_signature(
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

        previous_canonical_json = seen_state_signatures.get(state_signature)
        exact_repeated_state = (
            previous_canonical_json is not None
            and previous_canonical_json == state_canonical_json
        )

        if exact_repeated_state:
            return _build_result(
                converged=False,
                stop_reason=GlobalGoalCycleStopReason.REPEATED_STATE_DETECTED,
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                current_high_level_goals=current_high_level_goals,
                current_low_level_goals=current_low_level_goals,
                best_high_level_goals=best_high_level_goals,
                best_low_level_goals=best_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                bottom_up_errors=bottom_up_errors,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
            )

        # Semantic repeated-state detection: only checked once the state is
        # not an exact repeat. A state only counts as a semantic repeat when
        # BOTH the HLG/LLG collections are semantically equivalent (raw
        # text, via embeddings + cosine similarity) AND the branch decisions
        # are structurally equivalent (deterministic, rationale/free text
        # excluded) against the very same previously seen state. Comparing
        # HLG/LLG alone would misclassify a state whose decisions actually
        # improved (e.g. REGENERATE_LOW_LEVEL_GOALS -> CONFIRM_BRANCH) as a
        # stall, even though the cycle is still making progress.
        current_structural_decisions = _build_structural_decisions(evaluations)
        semantically_repeated_state = any(
            previous_structural_decisions == current_structural_decisions
            and states_semantically_equivalent(
                high_level_goals_a=current_high_level_goals,
                high_level_goals_b=previous_high_level_goals,
                low_level_goals_a=current_low_level_goals,
                low_level_goals_b=previous_low_level_goals,
            )
            for (
                previous_high_level_goals,
                previous_low_level_goals,
                previous_structural_decisions,
            ) in seen_states
        )

        if semantically_repeated_state:
            return _build_result(
                converged=False,
                stop_reason=(
                    GlobalGoalCycleStopReason.SEMANTIC_REPEATED_STATE_DETECTED
                ),
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                current_high_level_goals=current_high_level_goals,
                current_low_level_goals=current_low_level_goals,
                best_high_level_goals=best_high_level_goals,
                best_low_level_goals=best_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                bottom_up_errors=bottom_up_errors,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
            )

        seen_state_signatures[state_signature] = state_canonical_json
        seen_states.append((
            current_high_level_goals.model_copy(deep=True),
            current_low_level_goals.model_copy(deep=True),
            current_structural_decisions,
        ))

        if bottom_up_errors:
            return _build_result(
                converged=False,
                stop_reason=(
                    GlobalGoalCycleStopReason.BOTTOM_UP_RECONSTRUCTION_FAILED
                ),
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                current_high_level_goals=current_high_level_goals,
                current_low_level_goals=current_low_level_goals,
                best_high_level_goals=best_high_level_goals,
                best_low_level_goals=best_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                bottom_up_errors=bottom_up_errors,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
            )

        if evaluation_errors:
            return _build_result(
                converged=False,
                stop_reason=GlobalGoalCycleStopReason.GLOBAL_EVALUATION_FAILED,
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                current_high_level_goals=current_high_level_goals,
                current_low_level_goals=current_low_level_goals,
                best_high_level_goals=best_high_level_goals,
                best_low_level_goals=best_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
            )

        # Only a state that reached this point has completed bottom-up
        # reconstruction + global branch evaluation with no structural
        # errors: it is now eligible to become the best_validated_state.
        candidate_score = _compute_validated_state_quality(
            branch_map=branch_map,
            evaluations=evaluations,
            empty_branches=empty_branches,
        )

        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_high_level_goals = current_high_level_goals.model_copy(
                deep=True
            )
            best_low_level_goals = current_low_level_goals.model_copy(
                deep=True
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
                    stop_reason=(
                        GlobalGoalCycleStopReason
                        .DOCUMENTATION_COVERAGE_EVALUATION_FAILED
                    ),
                    completed_iterations=iteration_number,
                    max_iterations=max_iterations,
                    current_high_level_goals=current_high_level_goals,
                    current_low_level_goals=current_low_level_goals,
                    best_high_level_goals=best_high_level_goals,
                    best_low_level_goals=best_low_level_goals,
                    iterations=iteration_traces,
                    added_high_level_goals=all_added_high_level_goals,
                    coverage_error=coverage_error,
                )

            trace.documentation_coverage = coverage
            trace.documentation_fully_covered = (
                coverage.status == DocumentationCoverageStatus.COMPLETE
            )

            if coverage.status == DocumentationCoverageStatus.COMPLETE:
                return _build_result(
                    converged=True,
                    stop_reason=(
                        GlobalGoalCycleStopReason
                        .ALL_BRANCHES_CONFIRMED_AND_DOCUMENTATION_COVERED
                    ),
                    completed_iterations=iteration_number,
                    max_iterations=max_iterations,
                    current_high_level_goals=current_high_level_goals,
                    current_low_level_goals=current_low_level_goals,
                    best_high_level_goals=best_high_level_goals,
                    best_low_level_goals=best_low_level_goals,
                    iterations=iteration_traces,
                    added_high_level_goals=all_added_high_level_goals,
                )

            generation_requests = coverage.generation_requests
            trace.high_level_generation_requests = generation_requests

            (
                generated_by_request,
                duplicate_resolutions,
                high_level_error,
            ) = _generate_requested_high_level_goals(
                generate_high_level_goals=generate_high_level_goals,
                requests=generation_requests,
                existing_high_level_goals=branch_map,
                select_best_duplicate_high_level_goal=(
                    select_best_duplicate_high_level_goal
                ),
                project_description=project_description,
            )
            trace.high_level_duplicate_resolutions = duplicate_resolutions

            if high_level_error is not None:
                trace.high_level_regeneration_error = high_level_error
                return _build_result(
                    converged=False,
                    stop_reason=(
                        GlobalGoalCycleStopReason.HIGH_LEVEL_REGENERATION_FAILED
                    ),
                    completed_iterations=iteration_number,
                    max_iterations=max_iterations,
                    current_high_level_goals=current_high_level_goals,
                    current_low_level_goals=current_low_level_goals,
                    best_high_level_goals=best_high_level_goals,
                    best_low_level_goals=best_low_level_goals,
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
                        duplicate_resolutions=duplicate_resolutions,
                    )
                )
            except Exception as exc:
                high_level_error = f"{type(exc).__name__}: {exc}"
                trace.high_level_regeneration_error = high_level_error
                return _build_result(
                    converged=False,
                    stop_reason=(
                        GlobalGoalCycleStopReason.HIGH_LEVEL_REGENERATION_FAILED
                    ),
                    completed_iterations=iteration_number,
                    max_iterations=max_iterations,
                    current_high_level_goals=current_high_level_goals,
                    current_low_level_goals=current_low_level_goals,
                    best_high_level_goals=best_high_level_goals,
                    best_low_level_goals=best_low_level_goals,
                    iterations=iteration_traces,
                    added_high_level_goals=all_added_high_level_goals,
                    high_level_regeneration_error=high_level_error,
                )

            trace.newly_added_high_level_goals = added_goals
            trace.updated_high_level_goals = updated_high_level_goals
            trace.requires_regeneration = True
            all_added_high_level_goals.extend(added_goals)

            # Coverage-generated winners need a first LLG decomposition.
            # If a GENERATED HLG won against one or more EXISTING duplicates,
            # the losing existing branches have already been removed from the
            # HLG state by _append_coverage_generated_goals; remove their old
            # LLG decompositions as well. If an EXISTING HLG won, the losing
            # generated candidate is simply absent from added_goals.
            targets = HighLevelGoals(goals=added_goals)
            discarded_existing_branch_ids = {
                branch_id
                for resolution in duplicate_resolutions
                for branch_id in resolution.discarded_existing_branch_ids
            }
            replaced_parent_names = {
                normalize_goal_name(branch_map[branch_id].name)
                for branch_id in discarded_existing_branch_ids
                if branch_id in branch_map
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
                    stop_reason=(
                        GlobalGoalCycleStopReason.LOW_LEVEL_REGENERATION_FAILED
                    ),
                    completed_iterations=iteration_number,
                    max_iterations=max_iterations,
                    current_high_level_goals=current_high_level_goals,
                    current_low_level_goals=current_low_level_goals,
                    best_high_level_goals=best_high_level_goals,
                    best_low_level_goals=best_low_level_goals,
                    iterations=iteration_traces,
                    added_high_level_goals=all_added_high_level_goals,
                    evaluation_errors={"low_level_regeneration": error},
                )

            trace.regenerated_low_level_goals = merged

            # The coverage evaluator can occasionally propose an intention
            # that, after ordinary top-down HLG generation, resolves entirely
            # to HLGs already present in the state. If every generated result
            # is absorbed by an EXISTING winner and no existing branch loses,
            # the reported coverage gaps have been semantically reconciled.
            # Converge here instead of entering an identical next iteration
            # that would immediately trigger repeated-state detection.
            coverage_absorbed_by_existing_hlgs = (
                bool(generation_requests)
                and bool(duplicate_resolutions)
                and not added_goals
                and not discarded_existing_branch_ids
                and all(
                    not generated_by_request.get(request.request_id, [])
                    for request in generation_requests
                )
            )
            if coverage_absorbed_by_existing_hlgs:
                trace.documentation_fully_covered = True
                trace.requires_regeneration = False
                trace.updated_high_level_goals = current_high_level_goals
                trace.regenerated_low_level_goals = current_low_level_goals
                return _build_result(
                    converged=True,
                    stop_reason=(
                        GlobalGoalCycleStopReason
                        .ALL_BRANCHES_CONFIRMED_AND_DOCUMENTATION_COVERED
                    ),
                    completed_iterations=iteration_number,
                    max_iterations=max_iterations,
                    current_high_level_goals=current_high_level_goals,
                    current_low_level_goals=current_low_level_goals,
                    best_high_level_goals=best_high_level_goals,
                    best_low_level_goals=best_low_level_goals,
                    iterations=iteration_traces,
                    added_high_level_goals=all_added_high_level_goals,
                )

            current_high_level_goals = updated_high_level_goals
            current_low_level_goals = merged
            continue

        # Not all branches confirmed: collect every ADD/REPLACE generation
        # request emitted by the branch evaluations above (CONFIRM_BRANCH and
        # REGENERATE_LOW_LEVEL_GOALS branches contribute none).
        try:
            generation_requests = _collect_high_level_generation_requests(
                evaluations
            )
        except Exception as exc:
            high_level_error = f"{type(exc).__name__}: {exc}"
            trace.high_level_regeneration_error = high_level_error
            return _build_result(
                converged=False,
                stop_reason=(
                    GlobalGoalCycleStopReason.HIGH_LEVEL_REGENERATION_FAILED
                ),
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                current_high_level_goals=current_high_level_goals,
                current_low_level_goals=current_low_level_goals,
                best_high_level_goals=best_high_level_goals,
                best_low_level_goals=best_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
                high_level_regeneration_error=high_level_error,
            )

        trace.high_level_generation_requests = generation_requests

        # Run each request through the injected top-down generator callback
        # (with its own semantic retry and duplicate-cluster resolution);
        # this is the "generate" half of the branch-regen path, mirrored by
        # the identical call in the documentation-coverage branch above.
        (
            generated_by_request,
            duplicate_resolutions,
            high_level_error,
        ) = _generate_requested_high_level_goals(
            generate_high_level_goals=generate_high_level_goals,
            requests=generation_requests,
            existing_high_level_goals=branch_map,
            select_best_duplicate_high_level_goal=(
                select_best_duplicate_high_level_goal
            ),
            project_description=project_description,
        )
        trace.high_level_duplicate_resolutions = duplicate_resolutions

        if high_level_error is not None:
            trace.high_level_regeneration_error = high_level_error
            return _build_result(
                converged=False,
                stop_reason=(
                    GlobalGoalCycleStopReason.HIGH_LEVEL_REGENERATION_FAILED
                ),
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                current_high_level_goals=current_high_level_goals,
                current_low_level_goals=current_low_level_goals,
                best_high_level_goals=best_high_level_goals,
                best_low_level_goals=best_low_level_goals,
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
            # Apply ADD/REPLACE deterministically to get the next HLG
            # collection, then decide which branches need a *new* LLG
            # decomposition (added/replaced HLGs, plus branches whose
            # decision was REGENERATE_LOW_LEVEL_GOALS or
            # MATCHES_OTHER_HIGH_LEVEL_GOAL). Confirmed branches are excluded
            # from targets and keep their current LLGs untouched below.
            updated_high_level_goals, newly_added = (
                _apply_branch_high_level_generation(
                    existing_high_level_goals=branch_map,
                    requests=generation_requests,
                    generated_by_request=generated_by_request,
                    duplicate_resolutions=duplicate_resolutions,
                )
            )

            targets, replaced_parent_names = (
                _collect_branch_regeneration_targets(
                    branch_map=branch_map,
                    evaluations=evaluations,
                    generated_by_request=generated_by_request,
                    duplicate_resolutions=duplicate_resolutions,
                )
            )
        except Exception as exc:
            high_level_error = f"{type(exc).__name__}: {exc}"
            trace.high_level_regeneration_error = high_level_error
            return _build_result(
                converged=False,
                stop_reason=(
                    GlobalGoalCycleStopReason.HIGH_LEVEL_REGENERATION_FAILED
                ),
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                current_high_level_goals=current_high_level_goals,
                current_low_level_goals=current_low_level_goals,
                best_high_level_goals=best_high_level_goals,
                best_low_level_goals=best_low_level_goals,
                iterations=iteration_traces,
                added_high_level_goals=all_added_high_level_goals,
                evaluation_errors=evaluation_errors,
                empty_branches=empty_branches,
                high_level_regeneration_error=high_level_error,
            )

        trace.updated_high_level_goals = updated_high_level_goals
        trace.newly_added_high_level_goals = newly_added
        all_added_high_level_goals.extend(newly_added)

        # Regenerate LLGs only for `targets`; the callback never sees the
        # confirmed branches, and _regenerate_selected_branches merges the
        # fresh decomposition back with the untouched confirmed-branch LLGs.
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
                stop_reason=(
                    GlobalGoalCycleStopReason.LOW_LEVEL_REGENERATION_FAILED
                ),
                completed_iterations=iteration_number,
                max_iterations=max_iterations,
                current_high_level_goals=current_high_level_goals,
                current_low_level_goals=current_low_level_goals,
                best_high_level_goals=best_high_level_goals,
                best_low_level_goals=best_low_level_goals,
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

    # The loop ended naturally (MAX_ITERATIONS_REACHED) without an explicit
    # `if iteration_number == max_iterations` special case: whatever state
    # current_high_level_goals/current_low_level_goals holds here may be a
    # just-(re)generated candidate that never went through another
    # reconstruction + evaluation pass, so _build_result falls back to
    # best_validated_state instead of this raw candidate.
    return _build_result(
        converged=False,
        stop_reason=GlobalGoalCycleStopReason.MAX_ITERATIONS_REACHED,
        completed_iterations=max_iterations,
        max_iterations=max_iterations,
        current_high_level_goals=current_high_level_goals,
        current_low_level_goals=current_low_level_goals,
        best_high_level_goals=best_high_level_goals,
        best_low_level_goals=best_low_level_goals,
        iterations=iteration_traces,
        added_high_level_goals=all_added_high_level_goals,
        bottom_up_errors=last_bottom_up_errors,
        evaluation_errors=last_evaluation_errors,
        empty_branches=last_empty_branches,
    )
