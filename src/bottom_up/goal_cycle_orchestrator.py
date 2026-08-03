"""
goal_cycle_orchestrator.py

Deterministic orchestration of the bounded outer
refinement-abstraction-verification cycle.

The initial top-down pipeline is not executed by this module. The cycle starts
from already-generated HighLevelGoals and LowLevelGoals. During refinement it:
1. reconstructs high-level goal candidates bottom-up;
2. evaluates each branch;
3. preserves the low-level goals of confirmed branches;
4. regenerates only branches that require revision or newly added HLGs;
5. checks global documentation coverage when all branches are confirmed;
6. repeats until all branches are confirmed and coverage is complete.
"""

import hashlib
import json
from pathlib import Path
from typing import Callable

from src.data_model import (
    GlobalGoalCycleIteration,
    GlobalGoalCycleResult,
    DocumentationCoverageResult,
    GlobalGoalEvaluationResult,
    HighLevelGoal,
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


# The injected callback receives ONLY the HLGs whose LLGs must be generated.
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


def apply_global_evaluations(
    existing_high_level_goals: dict[str, HighLevelGoal],
    evaluations: dict[str, GlobalGoalEvaluationResult],
) -> HighLevelGoals:
    """Apply evaluator decisions to the HLG collection without calling an LLM."""
    updated_goals: list[HighLevelGoal] = []
    seen_normalized_names: set[str] = set()

    # Primo passaggio: per ogni HLG esistente, sostituiscilo con la
    # riscrittura proposta se la decisione lo richiede, altrimenti mantieni
    # l'originale.
    for branch_id, original in existing_high_level_goals.items():
        evaluation = evaluations.get(branch_id)

        if (
            evaluation is not None
            and evaluation.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL"
        ):
            resulting = evaluation.new_or_replacement_high_level_goal
            if resulting is None:
                raise ValueError(
                    f"Branch '{branch_id}' requires a replacement HLG but none exists."
                )
        else:
            resulting = original

        updated_goals.append(resulting)
        seen_normalized_names.add(normalize_goal_name(resulting.name))

    # Secondo passaggio: aggiungi i nuovi HLG proposti, evitando duplicati
    # (per nome normalizzato) rispetto a quelli già presenti.
    for evaluation in evaluations.values():
        if evaluation.decision != "ADD_NEW_HIGH_LEVEL_GOAL":
            continue

        new_goal = evaluation.new_or_replacement_high_level_goal
        if new_goal is None:
            raise ValueError(
                f"Branch '{evaluation.branch_id}' requires a new HLG but none exists."
            )

        normalized_name = normalize_goal_name(new_goal.name)
        if normalized_name in seen_normalized_names:
            continue

        seen_normalized_names.add(normalized_name)
        updated_goals.append(new_goal)

    return HighLevelGoals(goals=updated_goals)


def append_missing_high_level_goals(
    current_high_level_goals: HighLevelGoals,
    coverage_result: DocumentationCoverageResult,
) -> HighLevelGoals:
    """Apply validated documentation-coverage additions to the HLG collection."""
    return _deduplicate_goals(
        [
            *current_high_level_goals.goals,
            *coverage_result.added_high_level_goals,
        ]
    )


def _build_state_signature(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
    decisions: dict[str, object] | None = None,
) -> str:
    # Serializza lo stato corrente (HLG, LLG, decisioni) in modo canonico e
    # ne calcola l'hash: se lo stesso stato si ripresenta a un'iterazione
    # successiva, il ciclo è entrato in un loop e va interrotto.
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
    branch_map: dict[str, object],
    evaluations: dict[str, object],
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


def _collect_branch_regeneration_targets(
    branch_map: dict[str, HighLevelGoal],
    evaluations: dict[str, GlobalGoalEvaluationResult],
) -> tuple[HighLevelGoals, set[str]]:
    """
    Returns:
    - the HLGs whose low-level decomposition must be generated;
    - the normalized parent names whose existing LLGs must be removed.

    The evaluator JSON is complete, so every branch must have exactly one
    evaluation. Empty branches are represented through the deterministic
    REGENERATE_LOW_LEVEL_GOALS decision and require no special handling here.
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

    # Per ogni branch non confermato, decide quali HLG devono ricevere una
    # nuova decomposizione in base al tipo di decisione preso dal valutatore.
    for branch_id, original in branch_map.items():
        evaluation = evaluations[branch_id]

        if evaluation.decision == "CONFIRM_BRANCH":
            continue

        replaced_parent_names.add(
            normalize_goal_name(original.name)
        )

        if evaluation.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL":
            replacement = (
                evaluation.new_or_replacement_high_level_goal
            )
            if replacement is None:
                raise ValueError(
                    f"Branch '{branch_id}' requires a replacement HLG "
                    "but none exists."
                )

            targets.append(replacement)

        elif evaluation.decision == "ADD_NEW_HIGH_LEVEL_GOAL":
            targets.append(original)

            new_goal = evaluation.new_or_replacement_high_level_goal
            if new_goal is None:
                raise ValueError(
                    f"Branch '{branch_id}' requires a new HLG "
                    "but none exists."
                )

            targets.append(new_goal)

        elif evaluation.decision in {
            "REGENERATE_LOW_LEVEL_GOALS",
            "MATCHES_OTHER_HIGH_LEVEL_GOAL",
        }:
            targets.append(original)

        else:
            raise ValueError(
                f"Branch '{branch_id}' has unsupported evaluator "
                f"decision '{evaluation.decision}'."
            )

    return _deduplicate_goals(targets), replaced_parent_names


def _merge_selectively_regenerated_low_level_goals(
    current_low_level_goals: LowLevelGoals,
    regenerated_low_level_goals: LowLevelGoals,
    replaced_parent_names: set[str],
) -> LowLevelGoals:
    """Preserves confirmed branches and replaces only requested branches."""
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
    )


def run_global_goal_cycle(
    project_description: str,
    initial_high_level_goals: HighLevelGoals,
    initial_low_level_goals: LowLevelGoals,
    regenerate_low_level_goals: LowLevelGoalRegenerator,
    evaluation_output_directory: str | Path,
    max_iterations: int = 5,
) -> GlobalGoalCycleResult:
    """
    Executes only the added bottom-up/refinement pipeline.

    The initial top-down actors, HLGs and LLGs are treated as immutable input.
    Confirmed branches retain their original/current LLGs. The injected
    generator is called only with HLGs that require a new decomposition.

    Every evaluator result is obligatorily persisted as one complete JSON for
    all expected branches and loaded back before any decision is applied. The
    evaluator JSON is therefore the mandatory, validated boundary between the
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
        # Passo 1: ricostruzione bottom-up di un HLG candidato per ogni branch.
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

        # Passo 2: valutazione globale di ogni branch (LLM di confronto).
        evaluations, evaluation_errors = evaluate_all_branches(
            project_description=project_description,
            existing_high_level_goals=branch_map,
            reconstructed_high_level_goals=reconstructed_goals,
            empty_branches=empty_branches,
        )

        # Passo 3: persiste la valutazione su file e la ricarica, così che
        # il JSON validato sia l'unico confine tra valutatore e orchestratore.
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

        # Se lo stato è già stato visto in un'iterazione precedente, il
        # ciclo non sta convergendo: si interrompe per evitare un loop infinito.
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
            # Tutti i branch sono confermati: prima di dichiarare convergenza
            # si verifica che la documentazione sia coperta interamente dagli HLG.
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
            trace.newly_added_high_level_goals = coverage.added_high_level_goals

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

            updated_high_level_goals = append_missing_high_level_goals(
                current_high_level_goals,
                coverage,
            )
            all_added_high_level_goals.extend(coverage.added_high_level_goals)
            trace.updated_high_level_goals = updated_high_level_goals
            trace.requires_regeneration = True

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

            # Coverage additions create only new branches. Existing confirmed
            # branch decompositions are preserved unchanged.
            targets = HighLevelGoals(goals=coverage.added_high_level_goals)
            replaced_parent_names = {
                normalize_goal_name(goal.name)
                for goal in coverage.added_high_level_goals
            }
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

        # Non tutti i branch sono confermati: applica le decisioni del
        # valutatore agli HLG e rigenera selettivamente solo i branch necessari.
        updated_high_level_goals = apply_global_evaluations(
            existing_high_level_goals=branch_map,
            evaluations=evaluations,
        )
        trace.updated_high_level_goals = updated_high_level_goals

        already_recorded_names = {
            normalize_goal_name(goal.name)
            for goal in all_added_high_level_goals
        }
        for evaluation in evaluations.values():
            if evaluation.decision != "ADD_NEW_HIGH_LEVEL_GOAL":
                continue
            new_goal = evaluation.new_or_replacement_high_level_goal
            if new_goal is None:
                continue
            normalized_name = normalize_goal_name(new_goal.name)
            if normalized_name in already_recorded_names:
                continue
            already_recorded_names.add(normalized_name)
            all_added_high_level_goals.append(new_goal)

        targets, replaced_parent_names = _collect_branch_regeneration_targets(
            branch_map=branch_map,
            evaluations=evaluations,
        )

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
