"""Small bottom-up cycle with bounded LLG repair and global HLG discovery."""

from pathlib import Path

from src.bottom_up.global_goal_evaluator import (
    evaluate_all_branches,
    evaluate_missing_high_level_goals,
)
from src.bottom_up.goal_reconstructor import reconstruct_all_branches
from src.data_model import (
    Actor,
    Actors,
    BottomUpHighLevelGoal,
    GlobalGoalCycleIteration,
    GlobalGoalCycleResult,
    GlobalGoalCycleStopReason,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationResult,
    GoalBranch,
    HighLevelGoal,
    HighLevelGoalRemovalBasis,
    HighLevelGoalGenerationAction,
    HighLevelGoalGenerationRequest,
    HighLevelGoalGeneratorInput,
    HighLevelGoals,
    LowLevelGoal,
    LowLevelGoalRegenerationRequest,
    LowLevelGoals,
    MissingHighLevelGoalDecision,
    MissingHighLevelGoalEvaluation,
)
from src.examples.shot_learning import ShotPromptingMode
from src.extraction.extractor import (
    generate_high_level_goals as top_down_generate_high_level_goals,
    generate_low_level_goals as top_down_generate_low_level_goals,
)
from src.self_critique.refine_response import (
    EvalMode,
    generate_response_with_reflection,
)
from src.llm_clients import EvaluatorConversation


DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS = 5
MAX_LLG_REGENERATIONS_PER_BRANCH = 2
MAX_HLG_REWRITES_PER_BRANCH = 2
MAX_HLG_REMOVALS_PER_ITERATION = 2
MAX_REPLACEMENT_HLGS_PER_REWRITE = 1
BOTTOM_UP_CRITIC_MEMORY_TURNS = 4
BOTTOM_UP_CRITIC_STATE_UPDATES = 2


def _model_payload(value):
    """Return plain data so Pydantic can cross module-reload boundaries."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return value


def _coerce_model(value, model_type, *, context: str):
    """Rebuild a model with the schema class currently used by this module."""
    try:
        return model_type.model_validate(_model_payload(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid {context}: {exc}") from exc


def _key(value: str) -> str:
    return " ".join(value.casefold().split())


def _stable_hlg_key(goal: HighLevelGoal) -> str:
    """Identify a branch independently from its position in the HLG list."""
    return f"{_key(goal.actor.name)}::{_key(goal.name)}"


def _duplicate_hlg_keys(high_level_goals: HighLevelGoals) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for goal in high_level_goals.goals:
        goal_key = _stable_hlg_key(goal)
        if goal_key in seen:
            duplicates.add(goal_key)
        seen.add(goal_key)
    return duplicates


def _consolidate_duplicate_hlgs(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
) -> tuple[HighLevelGoals, LowLevelGoals, set[str]]:
    """Merge duplicate HLG identities and preserve all distinct child LLGs."""
    canonical_by_key: dict[str, HighLevelGoal] = {}
    canonical_goals: list[HighLevelGoal] = []
    duplicate_keys: set[str] = set()

    for goal in high_level_goals.goals:
        goal_key = _stable_hlg_key(goal)
        if goal_key in canonical_by_key:
            duplicate_keys.add(goal_key)
            continue
        canonical_by_key[goal_key] = goal
        canonical_goals.append(goal)

    if not duplicate_keys:
        return high_level_goals, low_level_goals, set()

    remapped_llgs = []
    for llg in low_level_goals.low_level_goals:
        canonical_parent = canonical_by_key[
            _stable_hlg_key(llg.high_level_associated)
        ]
        remapped_llgs.append(
            llg.model_copy(update={"high_level_associated": canonical_parent})
        )

    return (
        HighLevelGoals(goals=canonical_goals),
        LowLevelGoals(
            low_level_goals=_deduplicate_llg_candidates(remapped_llgs)
        ),
        duplicate_keys,
    )


def _prune_orphan_hlgs(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
) -> tuple[HighLevelGoals, list[HighLevelGoal]]:
    """Remove HLGs that have no LLG with that exact embedded parent."""
    parents_with_children = [
        goal.high_level_associated
        for goal in low_level_goals.low_level_goals
    ]
    orphaned = [
        goal for goal in high_level_goals.goals
        if goal not in parents_with_children
    ]
    if not orphaned:
        return high_level_goals, []

    return (
        HighLevelGoals(
            goals=[
                goal for goal in high_level_goals.goals
                if goal in parents_with_children
            ]
        ),
        orphaned,
    )


def _branches_from_embedded_parents(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
) -> list[GoalBranch]:
    """Build one branch per HLG, including HLGs with no generated children."""
    branches = [
        GoalBranch(
            branch_id=f"branch_{index:03d}",
            high_level_goal=goal,
            low_level_goals=[],
        )
        for index, goal in enumerate(high_level_goals.goals, start=1)
    ]
    for llg in low_level_goals.low_level_goals:
        parent = llg.high_level_associated
        branch = next(
            (item for item in branches if item.high_level_goal == parent),
            None,
        )
        if branch is None:
            raise ValueError(
                "An LLG references a parent that is absent from the current "
                f"HLG state: {parent.actor.name}::{parent.name}."
            )
        branch.low_level_goals.append(llg)
    return branches


def _validate_embedded_parents(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
) -> None:
    """Reject a baseline whose embedded LLG parents are not official HLGs."""
    unexpected = {
        f"{parent.actor.name}::{parent.name}"
        for parent in (
            goal.high_level_associated
            for goal in low_level_goals.low_level_goals
        )
        if parent not in high_level_goals.goals
    }
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(
            "The top-down baseline is inconsistent: these embedded LLG "
            f"parents are absent from structuredHighLevelGoals: {names}. "
            "Regenerate the top-down baseline before running bottom-up."
        )


def _embedded_parent_key(
    parent: HighLevelGoal,
    expected_parents: dict[str, HighLevelGoal],
) -> str | None:
    """Validate a generated embedded parent without canonicalizing it."""
    matches = [
        key for key, expected in expected_parents.items()
        if parent == expected
    ]
    return matches[0] if len(matches) == 1 else None


def _generation_request(
    request_id: str,
    action: HighLevelGoalGenerationAction,
    description: str,
    actor: Actor,
    rationale: str,
    target_branch_id: str | None = None,
) -> HighLevelGoalGenerationRequest:
    return HighLevelGoalGenerationRequest(
        request_id=request_id,
        action=action,
        generator_input=HighLevelGoalGeneratorInput(
            project_description=description,
            actors=Actors(actors=[actor]),
        ),
        rationale=rationale,
        target_branch_id=target_branch_id,
    )


def _actors_from_goals(high_level_goals: HighLevelGoals) -> Actors:
    actors = []
    seen = set()
    for goal in high_level_goals.goals:
        actor_key = _key(goal.actor.name)
        if actor_key not in seen:
            seen.add(actor_key)
            actors.append(goal.actor)
    return Actors(actors=actors)


def _merge_actors(*actor_sets: Actors) -> Actors:
    actors = []
    seen = set()
    for actor_set in actor_sets:
        for actor in actor_set.actors:
            actor_key = _key(actor.name)
            if actor_key not in seen:
                seen.add(actor_key)
                actors.append(actor)
    return Actors(actors=actors)


def _actor_for_request(request_actor: str, actors: Actors) -> Actor | None:
    wanted = _key(request_actor)
    for actor in actors.actors:
        if _key(actor.name) == wanted:
            return actor
    return None


def _should_add_missing_hlg(
    *,
    missing_hlg_evaluation: MissingHighLevelGoalEvaluation,
    llg_regenerations: dict[str, tuple[HighLevelGoal, str, list[LowLevelGoal]]],
    replacements: dict[str, list[HighLevelGoal]],
    removed_names: set[str],
) -> bool:
    """Missing HLGs must be discovered even when LLG repairs are underway.

    LLG improvements are useful, but they must not suppress the explicit project-
    wide signal that an actor-level HLG is absent from the current set. Branch-level
    rewrite/remove actions still take precedence because they can restructure the
    current HLGs before a project-level addition is proposed.
    """
    if missing_hlg_evaluation.decision != MissingHighLevelGoalDecision.FOUND:
        return False
    if replacements or removed_names:
        return False
    return True


def _generate_hlgs_with_top_down(
    request: HighLevelGoalGenerationRequest,
    mode: ShotPromptingMode,
    evaluator_ablation: bool,
) -> HighLevelGoals:
    focused_description = (
        "Evaluator rationale (use this as the primary correction guidance):\n"
        f"{request.rationale}\n\n"
        f"{request.generator_input.project_description}"
    )
    result, _, _ = generate_response_with_reflection(
        target_type="High Level Goals",
        call_function=top_down_generate_high_level_goals,
        define_args=(
            focused_description,
            request.generator_input.actors,
        ),
        eval_mode=EvalMode.HIGH_LEVEL,
        eval_args=(
            focused_description,
            request.generator_input.actors,
        ),
        shotPromptingMode=mode,
        llama_ablation=evaluator_ablation,
        focused_scope=(
            "Evaluate only the supplied actor and the single functional "
            "intention described by this replacement request."
        ),
    )
    result = _coerce_model(
        result,
        HighLevelGoals,
        context="High-Level Goal generator output",
    )
    if not result.goals:
        raise ValueError("The top-down pipeline generated no High-Level Goals.")
    allowed_actors = {
        _key(actor.name): actor
        for actor in request.generator_input.actors.actors
    }
    if not allowed_actors:
        raise ValueError("HLG generation requires at least one known actor.")
    normalized_goals = []
    for goal in result.goals:
        actor = allowed_actors.get(_key(goal.actor.name))
        if actor is None:
            if len(allowed_actors) != 1:
                raise ValueError(
                    f"Generated HLG '{goal.name}' uses unknown actor "
                    f"'{goal.actor.name}'."
                )
            actor = next(iter(allowed_actors.values()))
        normalized_goals.append(goal.model_copy(update={"actor": actor}))
    return HighLevelGoals(goals=normalized_goals)


def _regenerate_llgs_with_top_down(
    request: LowLevelGoalRegenerationRequest,
    mode: ShotPromptingMode,
    evaluator_ablation: bool,
) -> LowLevelGoals:
    actors = _actors_from_goals(request.high_level_goals)
    scope = "\n".join(
        f"- {goal.actor.name}: {goal.description}"
        for goal in request.high_level_goals.goals
    )
    feedback = "\n".join(request.guidance_by_parent_name.values())
    existing_by_parent: dict[str, list[LowLevelGoal]] = {}
    for llg in request.existing_low_level_goals.low_level_goals:
        existing_by_parent.setdefault(
            _stable_hlg_key(llg.high_level_associated), []
        ).append(llg)
    constraints = []
    for goal in request.high_level_goals.goals:
        parent_key = _stable_hlg_key(goal)
        existing = existing_by_parent.get(parent_key, [])
        constraints.append(
            f"- {goal.name}: keep valid existing goals and return the "
            "complete, non-redundant LLG decomposition at atomic functional "
            "interaction granularity; there is no fixed numeric cap. Keep "
            "distinct documented actions separate instead of collapsing them "
            f"into a generic operation. Existing LLGs: {[item.name for item in existing]}"
        )
    focused_description = (
        "Evaluator rationale (use this as the primary correction guidance):\n"
        f"{feedback or 'No additional rationale provided.'}\n\n"
        "Generate only the decomposition of these documented stakeholder "
        f"intentions:\n{scope}\n\n"
        "Repair constraints:\n"
        f"{chr(10).join(constraints)}\n"
        "Preserve already valid LLGs. Do not add generic CRUD variants, UI "
        "steps, notifications, confirmations, or sibling-HLG responsibilities "
        "unless the evaluator explicitly identifies them as essential. Cover "
        "every distinct operation explicitly required by the documentation, "
        "using a complete and non-redundant decomposition."
    )

    def generate_with_context(high_level_goals, feedback=None, mode=mode):
        return top_down_generate_low_level_goals(
            high_level_goals,
            feedback=feedback,
            mode=mode,
            generation_context=focused_description,
        )

    result, _, _ = generate_response_with_reflection(
        target_type="Low Level Goals",
        call_function=generate_with_context,
        define_args=(request.high_level_goals,),
        eval_mode=EvalMode.LOW_LEVEL,
        eval_args=(focused_description, actors, request.high_level_goals),
        shotPromptingMode=mode,
        llama_ablation=evaluator_ablation,
        focused_scope=(
            "Evaluate only the supplied actor, the listed parent HLGs, and "
            "their LLG decomposition. Do not require other project goals."
        ),
    )
    return _coerce_model(
        result,
        LowLevelGoals,
        context="Low-Level Goal generator output",
    )


def _deduplicate_llg_candidates(goals: list[LowLevelGoal]) -> list[LowLevelGoal]:
    """Remove exact semantic labels emitted more than once for one parent."""
    kept = []
    seen = set()
    for goal in goals:
        identity = (
            _stable_hlg_key(goal.high_level_associated),
            _key(goal.name),
            _key(goal.description),
        )
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(goal)
    return kept


def _accept_regenerated_llgs(
    *,
    existing: list[LowLevelGoal],
    generated: list[LowLevelGoal],
) -> tuple[list[LowLevelGoal], str | None]:
    """Reject empty replacements and preserve the old branch."""
    candidates = _deduplicate_llg_candidates(generated)
    preserved = list(existing)
    if not candidates:
        return preserved, "LLG_REGENERATION_REJECTED_EMPTY"
    return candidates, None


def _save_iteration(directory: str | Path | None, trace: GlobalGoalCycleIteration):
    if directory is None:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"iteration_{trace.iteration:03d}.json").write_text(
        trace.model_dump_json(indent=2), encoding="utf-8"
    )


def _with_decision(
    evaluation: GlobalGoalEvaluationResult,
    decision: GlobalGoalEvaluationDecision,
    rationale_suffix: str,
) -> GlobalGoalEvaluationResult:
    return evaluation.model_copy(
        update={
            "final_decision": decision,
            "rationale": f"{evaluation.rationale} {rationale_suffix}",
        }
    )


def _select_safe_hlg_removals(
    removal_branches: list[GoalBranch],
    evaluations: dict[str, GlobalGoalEvaluationResult],
) -> tuple[set[str], list[str]]:
    """Select a small removal batch while preserving every named cover HLG."""
    selected: set[str] = set()
    protected: set[str] = set()
    warnings: list[str] = []

    for branch in removal_branches:
        if len(selected) >= MAX_HLG_REMOVALS_PER_ITERATION:
            break
        branch_key = _stable_hlg_key(branch.high_level_goal)
        hlg_evaluation = evaluations[branch.branch_id].high_level_evaluation
        if branch_key in protected:
            warnings.append(
                f"{branch.branch_id}: HLG_REMOVAL_SKIPPED_COVER_TARGET"
            )
            continue

        if (
            hlg_evaluation.removal_basis
            == HighLevelGoalRemovalBasis.FULLY_REDUNDANT
        ):
            target_key = (
                f"{_key(branch.high_level_goal.actor.name)}::"
                f"{_key(hlg_evaluation.covered_by_high_level_goal_name or '')}"
            )
            if target_key in selected:
                warnings.append(
                    f"{branch.branch_id}: HLG_REMOVAL_SKIPPED_CIRCULAR_COVER"
                )
                continue
            protected.add(target_key)

        selected.add(branch_key)

    if len(removal_branches) > len(selected):
        warnings.append(
            "HLG_REMOVAL_BATCH_LIMIT_OR_DEPENDENCY_GUARD "
            f"({len(removal_branches)} requested, {len(selected)} applied)"
        )
    return selected, warnings


def _llg_names_by_parent(
    low_level_goals: LowLevelGoals,
) -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for goal in low_level_goals.low_level_goals:
        parent_key = _stable_hlg_key(goal.high_level_associated)
        names.setdefault(parent_key, set()).add(goal.name)
    return names


def _critic_state_summary(
    *,
    iteration: int,
    previous_hlgs: HighLevelGoals,
    previous_llgs: LowLevelGoals,
    current_hlgs: HighLevelGoals,
    current_llgs: LowLevelGoals,
) -> str:
    previous_keys = {_stable_hlg_key(goal) for goal in previous_hlgs.goals}
    current_keys = {_stable_hlg_key(goal) for goal in current_hlgs.goals}
    added = sorted(current_keys - previous_keys)
    removed = sorted(previous_keys - current_keys)

    previous_llg_names = _llg_names_by_parent(previous_llgs)
    current_llg_names = _llg_names_by_parent(current_llgs)
    llg_changes = []
    for key in sorted(previous_llg_names.keys() | current_llg_names.keys()):
        added_llgs = sorted(
            current_llg_names.get(key, set())
            - previous_llg_names.get(key, set())
        )
        removed_llgs = sorted(
            previous_llg_names.get(key, set())
            - current_llg_names.get(key, set())
        )
        if added_llgs or removed_llgs:
            llg_changes.append(
                f"{key}: added {added_llgs or ['none']}, "
                f"removed {removed_llgs or ['none']}"
            )
    return (
        f"Iteration {iteration} applied hierarchy updates. "
        f"HLGs added: {added or ['none']}. "
        f"HLGs removed or replaced: {removed or ['none']}. "
        f"LLG changes: {llg_changes or ['none']}. "
        f"Current HLGs: {sorted(current_keys)}."
    )


def run_global_goal_cycle(
    *,
    project_description: str,
    initial_high_level_goals: HighLevelGoals,
    initial_low_level_goals: LowLevelGoals,
    mode: ShotPromptingMode = ShotPromptingMode.ZERO_SHOT,
    evaluator_ablation: bool = False,
    evaluation_output_directory: str | Path | None = None,
    max_iterations: int = DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS,
) -> GlobalGoalCycleResult:
    initial_high_level_goals = _coerce_model(
        initial_high_level_goals,
        HighLevelGoals,
        context="initial High-Level Goals",
    )
    initial_low_level_goals = _coerce_model(
        initial_low_level_goals,
        LowLevelGoals,
        context="initial Low-Level Goals",
    )
    _validate_embedded_parents(
        initial_high_level_goals,
        initial_low_level_goals,
    )
    current_hlgs = initial_high_level_goals.model_copy(deep=True)
    current_llgs = initial_low_level_goals.model_copy(deep=True)
    critic_conversation = EvaluatorConversation(
        max_turns=BOTTOM_UP_CRITIC_MEMORY_TURNS,
        max_state_updates=BOTTOM_UP_CRITIC_STATE_UPDATES,
    )
    critic_conversation.remember_state(
        "Initial input before iteration 1; no previous iteration exists. "
        "Current HLGs: "
        f"{sorted(_stable_hlg_key(goal) for goal in current_hlgs.goals)}. "
        f"Total LLGs: {len(current_llgs.low_level_goals)}."
    )
    llg_regeneration_counts: dict[str, int] = {}
    confirmed_branch_keys: set[str] = set()
    hlg_rewrite_counts: dict[str, int] = {}
    retired_hlgs: dict[str, HighLevelGoal] = {}
    traces: list[GlobalGoalCycleIteration] = []
    warnings: list[str] = []
    known_actors = _actors_from_goals(initial_high_level_goals)

    for iteration in range(1, max_iterations + 1):
        iteration_start_hlgs = current_hlgs.model_copy(deep=True)
        iteration_start_llgs = current_llgs.model_copy(deep=True)
        iteration_warnings: list[str] = []
        print(
            f"[bottom-up] iteration {iteration}/{max_iterations} starting",
            flush=True,
        )
        current_hlgs, current_llgs, consolidated_duplicate_keys = (
            _consolidate_duplicate_hlgs(current_hlgs, current_llgs)
        )
        if consolidated_duplicate_keys:
            confirmed_branch_keys.clear()
            iteration_warnings.append(
                "DUPLICATE_HIGH_LEVEL_GOALS_CONSOLIDATED: "
                + ", ".join(sorted(consolidated_duplicate_keys))
            )
        current_hlgs, orphaned_hlgs = _prune_orphan_hlgs(
            current_hlgs,
            current_llgs,
        )
        if orphaned_hlgs:
            for goal in orphaned_hlgs:
                retired_hlgs[_stable_hlg_key(goal)] = goal
            confirmed_branch_keys.clear()
            iteration_warnings.append(
                "ORPHAN_HIGH_LEVEL_GOALS_REMOVED: "
                + ", ".join(
                    f"{goal.actor.name}::{goal.name}"
                    for goal in orphaned_hlgs
                )
            )
        branches = _branches_from_embedded_parents(current_hlgs, current_llgs)

        # A confirmed branch is skipped while the HLG structure is unchanged.
        # Structural changes clear this set after the iteration and trigger a
        # fresh audit. Stable actor/name identity survives branch renumbering.
        active_branches = [
            branch
            for branch in branches
            if _stable_hlg_key(branch.high_level_goal)
            not in confirmed_branch_keys
        ]

        raw_reconstructions = reconstruct_all_branches(active_branches)
        reconstructions = {
            branch_id: _coerce_model(
                reconstruction,
                BottomUpHighLevelGoal,
                context=f"reconstruction for {branch_id}",
            )
            for branch_id, reconstruction in raw_reconstructions.items()
        }
        raw_evaluations = evaluate_all_branches(
            project_description,
            active_branches,
            reconstructions,
            current_hlgs,
            critic_conversation,
        )
        evaluations = {
            branch_id: _coerce_model(
                evaluation,
                GlobalGoalEvaluationResult,
                context=f"evaluation for {branch_id}",
            )
            for branch_id, evaluation in raw_evaluations.items()
        }
        confirmed_branch_keys.update(
            _stable_hlg_key(branch.high_level_goal)
            for branch in active_branches
            if evaluations[branch.branch_id].final_decision
            == GlobalGoalEvaluationDecision.CONFIRM_BRANCH
        )

        known_actors = _merge_actors(
            known_actors, _actors_from_goals(current_hlgs)
        )
        missing_hlg_evaluation = _coerce_model(
            evaluate_missing_high_level_goals(
                project_description,
                current_hlgs,
                known_actors,
                critic_conversation,
                HighLevelGoals(goals=list(retired_hlgs.values())),
            ),
            MissingHighLevelGoalEvaluation,
            context="missing High-Level Goal evaluation",
        )
        if (
            missing_hlg_evaluation.decision
            == MissingHighLevelGoalDecision.INCONCLUSIVE
        ):
            iteration_warnings.append(
                "MISSING_HLG_EVALUATION_INCONCLUSIVE"
            )

        removed_names: set[str] = set()
        llg_regenerations: dict[
            str, tuple[HighLevelGoal, str, list[LowLevelGoal]]
        ] = {}
        replacements: dict[str, list[HighLevelGoal]] = {}
        additions: list[HighLevelGoal] = []
        # Deterministic action order: one rewrite, then a bounded batch of
        # independent removals, then one LLG regeneration. Rewrites and LLG
        # repairs stay sequential because each can change another branch's
        # scope; removals were all evaluated against the same immutable state.
        rewrite_handled = False
        rewrite_branches = [
            branch for branch in active_branches
            if evaluations[branch.branch_id].final_decision
            == GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
        ]
        for branch in rewrite_branches:
            evaluation = evaluations[branch.branch_id]
            original_key = _stable_hlg_key(branch.high_level_goal)
            sibling_goals_by_key = {
                _stable_hlg_key(goal): goal
                for goal in current_hlgs.goals
                if _stable_hlg_key(goal) != original_key
            }
            rewrite_count = hlg_rewrite_counts.get(original_key, 0)
            if rewrite_count >= MAX_HLG_REWRITES_PER_BRANCH:
                evaluations[branch.branch_id] = _with_decision(
                    evaluation,
                    GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE,
                    "HLG rewrite limit reached; retaining the current HLG.",
                )
                iteration_warnings.append(
                    f"{branch.branch_id}: HLG_REWRITE_LIMIT_REACHED"
                )
                continue

            request = evaluation.replacement_request
            generated = _generate_hlgs_with_top_down(
                _generation_request(
                    f"{branch.branch_id}_rewrite",
                    HighLevelGoalGenerationAction.REPLACE_EXISTING_HIGH_LEVEL_GOAL,
                    request.generation_project_description,
                    request.actor,
                    request.rationale,
                    branch.branch_id,
                ),
                mode,
                evaluator_ablation,
            )
            replacement_goals = []
            rewrite_consolidated = False
            # A focused rewrite request yields one candidate at most;
            # accepting every candidate here caused HLG proliferation.
            for goal in generated.goals[:MAX_REPLACEMENT_HLGS_PER_REWRITE]:
                if not goal.name.strip():
                    continue
                goal_key = _stable_hlg_key(goal)
                existing_sibling = sibling_goals_by_key.get(goal_key)
                if existing_sibling is not None:
                    sibling_llgs = [
                        llg for llg in current_llgs.low_level_goals
                        if _stable_hlg_key(llg.high_level_associated) == goal_key
                    ]
                    replacements[original_key] = []
                    llg_regenerations[goal_key] = (
                        existing_sibling,
                        (
                            f"{request.rationale} The rewrite resolved to the "
                            "existing sibling HLG; preserve the documented "
                            "capabilities from both decompositions while "
                            "returning one non-redundant LLG set."
                        ),
                        [*sibling_llgs, *branch.low_level_goals],
                    )
                    rewrite_consolidated = True
                    iteration_warnings.append(
                        f"{branch.branch_id}: DUPLICATE_REWRITE_CONSOLIDATED_INTO "
                        f"'{existing_sibling.name}'"
                    )
                    break
                replacement_goals.append(goal)
                llg_regenerations[goal_key] = (
                    goal,
                    request.rationale,
                    list(branch.low_level_goals),
                )
            if replacement_goals or rewrite_consolidated:
                hlg_rewrite_counts[original_key] = rewrite_count + 1
                if replacement_goals:
                    replacements[original_key] = replacement_goals
            else:
                iteration_warnings.append(
                    f"{branch.branch_id}: HLG_REGENERATION_EMPTY_IGNORED"
                )
            rewrite_handled = True
            break

        if (
            not rewrite_handled
            and missing_hlg_evaluation.decision
            != MissingHighLevelGoalDecision.FOUND
        ):
            removal_branches = [
                branch for branch in active_branches
                if evaluations[branch.branch_id].final_decision
                == GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
            ]
            removed_names, removal_warnings = _select_safe_hlg_removals(
                removal_branches,
                evaluations,
            )
            iteration_warnings.extend(removal_warnings)

        if not rewrite_handled and not removed_names:
            regeneration_branches = [
                branch for branch in active_branches
                if evaluations[branch.branch_id].final_decision
                == GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS
            ]
            for branch in regeneration_branches:
                # Missing/redundant HLGs are repaired before decompositions so
                # LLG churn cannot starve the project-wide HLG pass.
                if (
                    missing_hlg_evaluation.decision
                    == MissingHighLevelGoalDecision.FOUND
                ):
                    break
                evaluation = evaluations[branch.branch_id]
                stable_key = _stable_hlg_key(branch.high_level_goal)
                count = llg_regeneration_counts.get(stable_key, 0)
                if count >= MAX_LLG_REGENERATIONS_PER_BRANCH:
                    evaluations[branch.branch_id] = _with_decision(
                        evaluation,
                        GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED,
                        "LLG regeneration limit reached; keeping the latest LLGs.",
                    )
                    llg_evaluation = evaluation.low_level_evaluation
                    unsupported = (
                        llg_evaluation.unsupported_or_misleading_llg_ids
                        if llg_evaluation else []
                    )
                    missing = (
                        llg_evaluation.missing_essential_capabilities
                        if llg_evaluation else []
                    )
                    iteration_warnings.append(
                        f"{branch.branch_id}: LLG_REGENERATION_LIMIT_REACHED "
                        f"(unsupported={unsupported or ['none']}; "
                        f"missing={missing or ['none']})"
                    )
                    continue
                llg_regeneration_counts[stable_key] = count + 1
                llg_regenerations[stable_key] = (
                    branch.high_level_goal,
                    evaluation.low_level_evaluation.regeneration_feedback
                    or evaluation.rationale,
                    list(branch.low_level_goals),
                )
                break

        unresolved_decisions = {
            evaluation.final_decision
            for evaluation in evaluations.values()
            if evaluation.final_decision
            in {
                GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED,
                GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE,
            }
        }
        if GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED in unresolved_decisions:
            iteration_warnings.append(
                "UNRESOLVED_LLG_REGENERATION_LIMIT_REACHED"
            )
        if GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE in unresolved_decisions:
            iteration_warnings.append("UNRESOLVED_EVALUATION_INCONCLUSIVE")

        # Apply branch-level HLG changes before the one global coverage check.
        next_goals: list[HighLevelGoal] = []
        for goal in current_hlgs.goals:
            key = _stable_hlg_key(goal)
            if key in removed_names:
                retired_hlgs[key] = goal
                continue
            if key in replacements:
                retired_hlgs[key] = goal
            next_goals.extend(replacements.get(key, [goal]))
        changed_parent_names = removed_names | set(replacements) | set(llg_regenerations)
        next_llgs = [
            llg
            for llg in current_llgs.low_level_goals
            if _stable_hlg_key(llg.high_level_associated) not in changed_parent_names
        ]

        # The branch evaluator can request LLG repair, but the global HLG
        # coverage check is deliberately a separate, single project-wide call.
        if llg_regenerations:
            target_goals = [item[0] for item in llg_regenerations.values()]
            guidance = {
                _stable_hlg_key(item[0]): item[1]
                for item in llg_regenerations.values()
            }
            existing_llgs = [
                llg
                for item in llg_regenerations.values()
                for llg in item[2]
            ]
            generated_llgs = _regenerate_llgs_with_top_down(
                LowLevelGoalRegenerationRequest(
                    high_level_goals=HighLevelGoals(goals=target_goals),
                    guidance_by_parent_name=guidance,
                    existing_low_level_goals=LowLevelGoals(
                        low_level_goals=existing_llgs
                    ),
                ),
                mode,
                evaluator_ablation,
            )
            parents = {_stable_hlg_key(goal): goal for goal in target_goals}
            generated_by_parent: dict[str, list[LowLevelGoal]] = {
                key: [] for key in parents
            }
            for llg in generated_llgs.low_level_goals:
                parent_key = _embedded_parent_key(
                    llg.high_level_associated, parents
                )
                if parent_key is None:
                    iteration_warnings.append(
                        f"LLG '{llg.name}' did not preserve an exact generated "
                        "parent HLG; skipped."
                    )
                    continue
                generated_by_parent[parent_key].append(llg)

            for parent_key, item in llg_regenerations.items():
                parent, _, existing = item
                accepted, rejection = _accept_regenerated_llgs(
                    existing=existing,
                    generated=generated_by_parent.get(parent_key, []),
                )
                next_llgs.extend(accepted)
                if rejection:
                    iteration_warnings.append(
                        f"{parent.name}: {rejection}"
                    )

        current_hlgs = HighLevelGoals(goals=next_goals)
        current_llgs = LowLevelGoals(low_level_goals=next_llgs)

        known_actors = _merge_actors(
            known_actors, _actors_from_goals(current_hlgs)
        )
        actors = known_actors

        if _should_add_missing_hlg(
            missing_hlg_evaluation=missing_hlg_evaluation,
            llg_regenerations=llg_regenerations,
            replacements=replacements,
            removed_names=removed_names,
        ):
            existing_goal_keys = {
                _stable_hlg_key(goal) for goal in current_hlgs.goals
            }
            for request_index, request in enumerate(
                missing_hlg_evaluation.missing_goal_requests,
                start=1,
            ):
                actor = _actor_for_request(request.actor, actors)
                if actor is None:
                    iteration_warnings.append(
                        f"{request.actor}: UNKNOWN_ACTOR_REQUEST_IGNORED"
                    )
                    continue
                generated = _generate_hlgs_with_top_down(
                    _generation_request(
                        f"iteration_{iteration}_missing_{request_index}",
                        HighLevelGoalGenerationAction.ADD_NEW_HIGH_LEVEL_GOAL,
                        request.generation_project_description,
                        actor,
                        missing_hlg_evaluation.rationale,
                    ),
                    mode,
                    evaluator_ablation,
                )
                new_goals = []
                for goal in generated.goals[:1]:
                    if not goal.name.strip():
                        continue
                    goal_key = _stable_hlg_key(goal)
                    if goal_key in existing_goal_keys:
                        iteration_warnings.append(
                            f"{goal.name}: DUPLICATE_DISCOVERED_HLG_IGNORED"
                        )
                        continue
                    new_goals.append(goal)
                    additions.append(goal)
                    existing_goal_keys.add(goal_key)
                    retired_hlgs.pop(goal_key, None)

            if additions:
                current_hlgs = HighLevelGoals(
                    goals=[*current_hlgs.goals, *additions]
                )
                generated_llgs = _regenerate_llgs_with_top_down(
                    LowLevelGoalRegenerationRequest(
                        high_level_goals=HighLevelGoals(goals=additions),
                        guidance_by_parent_name={
                            _stable_hlg_key(goal): missing_hlg_evaluation.rationale
                            for goal in additions
                        },
                    ),
                    mode,
                    evaluator_ablation,
                )
                parents = {_stable_hlg_key(goal): goal for goal in additions}
                generated_by_parent = {
                    _stable_hlg_key(goal): [] for goal in additions
                }
                for llg in generated_llgs.low_level_goals:
                    parent_key = _embedded_parent_key(
                        llg.high_level_associated, parents
                    )
                    if parent_key is not None:
                        generated_by_parent[parent_key].append(llg)
                    else:
                        iteration_warnings.append(
                            f"LLG '{llg.name}' did not preserve an exact "
                            "generated parent HLG; skipped."
                        )
                valid_additions = []
                for parent in additions:
                    accepted, rejection = _accept_regenerated_llgs(
                        existing=[],
                        generated=generated_by_parent.get(_stable_hlg_key(parent), []),
                    )
                    if accepted:
                        valid_additions.append(parent)
                        current_llgs.low_level_goals.extend(accepted)
                    if rejection:
                        iteration_warnings.append(
                            f"{parent.name}: {rejection}"
                        )
                if len(valid_additions) != len(additions):
                    valid_keys = {_stable_hlg_key(goal) for goal in valid_additions}
                    current_hlgs = HighLevelGoals(goals=[
                        goal for goal in current_hlgs.goals
                        if goal not in additions or _stable_hlg_key(goal) in valid_keys
                    ])

        # An HLG-level change alters the project-wide responsibility map. Audit
        # every branch again on the next iteration instead of trusting a KEEP
        # decision made against the previous HLG set.
        if replacements or removed_names or additions:
            confirmed_branch_keys.clear()

        critic_conversation.remember_state(
            _critic_state_summary(
                iteration=iteration,
                previous_hlgs=iteration_start_hlgs,
                previous_llgs=iteration_start_llgs,
                current_hlgs=current_hlgs,
                current_llgs=current_llgs,
            )
        )

        # Only real confirmations permit convergence. An evaluator failure is
        # transient and is retried on the next iteration; limit-related
        # inconclusive decisions remain terminal warnings.
        retryable_branch_inconclusive = any(
            evaluation.final_decision
            == GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE
            and "evaluation failed:" in evaluation.rationale.casefold()
            for evaluation in evaluations.values()
        )
        retryable_coverage_inconclusive = (
            missing_hlg_evaluation.decision
            == MissingHighLevelGoalDecision.INCONCLUSIVE
        )
        llg_parent_keys = {
            _stable_hlg_key(goal.high_level_associated)
            for goal in current_llgs.low_level_goals
        }
        orphan_hlgs = [
            goal for goal in current_hlgs.goals
            if _stable_hlg_key(goal) not in llg_parent_keys
        ]
        if orphan_hlgs:
            iteration_warnings.append(
                "HLGS_WITHOUT_LOW_LEVEL_GOALS: "
                + ", ".join(
                    f"{goal.actor.name}::{goal.name}" for goal in orphan_hlgs
                )
            )
        duplicate_hlg_keys = _duplicate_hlg_keys(current_hlgs)
        if duplicate_hlg_keys:
            iteration_warnings.append(
                "DUPLICATE_HIGH_LEVEL_GOALS: "
                + ", ".join(sorted(duplicate_hlg_keys))
            )
        all_confirmed = all(
            evaluation.final_decision
            == GlobalGoalEvaluationDecision.CONFIRM_BRANCH
            for evaluation in evaluations.values()
        ) and not orphan_hlgs and not duplicate_hlg_keys
        no_missing = (
            missing_hlg_evaluation.decision
            == MissingHighLevelGoalDecision.NO_MISSING
        )

        trace = GlobalGoalCycleIteration(
            iteration=iteration,
            reconstructions=reconstructions,
            evaluations=evaluations,
            high_level_goals=current_hlgs,
            low_level_goals=current_llgs,
            missing_hlg_evaluation=missing_hlg_evaluation,
            missing_hlg_decision=(
                GlobalGoalEvaluationDecision.MISSING_HIGH_LEVEL_GOALS_FOUND
                if missing_hlg_evaluation.decision
                == MissingHighLevelGoalDecision.FOUND
                else (
                    GlobalGoalEvaluationDecision.NO_MISSING_HIGH_LEVEL_GOALS
                    if missing_hlg_evaluation.decision
                    == MissingHighLevelGoalDecision.NO_MISSING
                    else GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE
                )
            ),
            llg_regeneration_counts=dict(llg_regeneration_counts),
            warnings=list(iteration_warnings),
        )
        traces.append(trace)
        _save_iteration(evaluation_output_directory, trace)
        warnings.extend(iteration_warnings)

        if all_confirmed and no_missing:
            return GlobalGoalCycleResult(
                converged=True,
                stop_reason=GlobalGoalCycleStopReason.ALL_BRANCHES_CONFIRMED,
                completed_iterations=iteration,
                final_high_level_goals=current_hlgs,
                final_low_level_goals=current_llgs,
                iterations=traces,
                llg_regeneration_counts=dict(llg_regeneration_counts),
                warnings=warnings,
            )

        actions_remain = bool(
            missing_hlg_evaluation.decision == MissingHighLevelGoalDecision.FOUND
            or retryable_branch_inconclusive
            or retryable_coverage_inconclusive
            or any(
                evaluation.final_decision
                in {
                    GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS,
                    GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL,
                    GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL,
                }
                for evaluation in evaluations.values()
            )
        )
        if no_missing and not actions_remain and iteration_warnings:
            return GlobalGoalCycleResult(
                converged=False,
                stop_reason=GlobalGoalCycleStopReason.NO_ACTIONS_REMAIN_WITH_WARNINGS,
                completed_iterations=iteration,
                final_high_level_goals=current_hlgs,
                final_low_level_goals=current_llgs,
                iterations=traces,
                llg_regeneration_counts=dict(llg_regeneration_counts),
                warnings=warnings,
            )

    return GlobalGoalCycleResult(
        converged=False,
        stop_reason=GlobalGoalCycleStopReason.MAX_ITERATIONS_REACHED,
        completed_iterations=max_iterations,
        final_high_level_goals=current_hlgs,
        final_low_level_goals=current_llgs,
        iterations=traces,
        llg_regeneration_counts=dict(llg_regeneration_counts),
        warnings=warnings,
    )
