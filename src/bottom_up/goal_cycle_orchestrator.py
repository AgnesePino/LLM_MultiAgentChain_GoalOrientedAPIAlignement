"""Small bottom-up cycle with bounded LLG repair and global HLG discovery."""

from pathlib import Path

from src.bottom_up.global_goal_evaluator import (
    evaluate_all_branches,
    evaluate_missing_high_level_goals,
)
from src.bottom_up.goal_reconstructor import reconstruct_all_branches
from src.bottom_up.low_level_goal_mapper import group_low_level_goals
from src.data_model import (
    Actor,
    Actors,
    GlobalGoalCycleIteration,
    GlobalGoalCycleResult,
    GlobalGoalCycleStopReason,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationResult,
    HighLevelGoal,
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
MAX_HLG_REMOVALS_PER_ITERATION = 5
MAX_NEW_HLGS_PER_ITERATION = 2
MAX_REPLACEMENT_HLGS_PER_REWRITE = 1
MAX_LLG_GROWTH_PER_REPAIR = 2
MAX_LLGS_FOR_NEW_HLG = 6
BOTTOM_UP_CRITIC_MEMORY_TURNS = 4
BOTTOM_UP_CRITIC_STATE_UPDATES = 2


def _key(value: str) -> str:
    return " ".join(value.casefold().split())


def _stable_hlg_key(goal: HighLevelGoal) -> str:
    """Identify a branch independently from its position in the HLG list."""
    return f"{_key(goal.actor.name)}::{_key(goal.name)}"


def _resolve_generated_parent(
    generated_parent: HighLevelGoal,
    parents: dict[str, HighLevelGoal],
) -> HighLevelGoal | None:
    """Resolve an LLM parent reference, tolerating actor wording drift.

    Generated LLGs sometimes preserve the HLG name but paraphrase its actor.
    An exact actor/name match wins; a name-only fallback is safe only when the
    requested parent name is unique.
    """
    exact = parents.get(_stable_hlg_key(generated_parent))
    if exact is not None:
        return exact
    matches = [
        parent
        for parent in parents.values()
        if _key(parent.name) == _key(generated_parent.name)
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


def _replace_parent(llg: LowLevelGoal, parent: HighLevelGoal) -> LowLevelGoal:
    return llg.model_copy(update={"high_level_associated": parent})


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
    llg_regenerations: dict[str, tuple[HighLevelGoal, str, list[LowLevelGoal], int]],
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


def _deduplicate_high_level_goals(
    goals: HighLevelGoals,
    llgs: LowLevelGoals,
    warnings: list[str],
) -> tuple[HighLevelGoals, LowLevelGoals]:
    """Remove conservative name/actor duplicates after one state change."""
    kept: list[HighLevelGoal] = []
    aliases: dict[str, HighLevelGoal] = {}
    for goal in goals.goals:
        key = _stable_hlg_key(goal)
        duplicate = next(
            (item for item in kept
             if _key(item.actor.name) == _key(goal.actor.name)
             and (_key(item.name) == key or key in _key(item.name) or _key(item.name) in key)),
            None,
        )
        if duplicate is not None:
            aliases[key] = duplicate
            warnings.append(f"HLG_DUPLICATE_REMOVED: {goal.name}")
        else:
            kept.append(goal)
            aliases[key] = goal

    remapped: list[LowLevelGoal] = []
    for llg in llgs.low_level_goals:
        parent = aliases.get(_stable_hlg_key(llg.high_level_associated))
        if parent is not None:
            remapped.append(_replace_parent(llg, parent))
    return HighLevelGoals(goals=kept), LowLevelGoals(low_level_goals=remapped)


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
    if not isinstance(result, HighLevelGoals) or not result.goals:
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
        maximum = request.max_goals_by_parent_name.get(
            parent_key, request.max_goals_by_parent_name.get(
                goal.name, MAX_LLGS_FOR_NEW_HLG
            )
        )
        constraints.append(
            f"- {goal.name}: keep valid existing goals; return at most "
            f"{maximum} LLGs total. Existing LLGs: "
            f"{[item.name for item in existing]}"
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
        "unless the evaluator explicitly identifies them as essential. Prefer "
        "the smallest complete decomposition."
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
    if not isinstance(result, LowLevelGoals):
        raise TypeError("The top-down pipeline did not return LowLevelGoals.")
    return result


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
    parent: HighLevelGoal,
    existing: list[LowLevelGoal],
    generated: list[LowLevelGoal],
    maximum: int,
) -> tuple[list[LowLevelGoal], str | None]:
    """Reject empty or proliferating replacements and preserve the old branch."""
    candidates = _deduplicate_llg_candidates(generated)
    preserved = [_replace_parent(item, parent) for item in existing]
    if not candidates:
        return preserved, "LLG_REGENERATION_REJECTED_EMPTY"
    if len(candidates) > maximum:
        if not existing:
            return (
                [_replace_parent(item, parent) for item in candidates[:maximum]],
                "LLG_REGENERATION_TRUNCATED_GOAL_GROWTH "
                f"({len(candidates)} generated, maximum {maximum})",
            )
        return preserved, (
            "LLG_REGENERATION_REJECTED_GOAL_GROWTH "
            f"({len(candidates)} generated, maximum {maximum})"
        )
    return [_replace_parent(item, parent) for item in candidates], None


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
        f"Iteration {iteration} applied state. "
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
    stabilized_llg_keys: set[str] = set()
    confirmed_branch_keys: set[str] = set()
    hlg_rewrite_counts: dict[str, int] = {}
    excluded_hlg_keys: set[str] = set()
    traces: list[GlobalGoalCycleIteration] = []
    warnings: list[str] = []
    known_actors = _actors_from_goals(initial_high_level_goals)

    for iteration in range(1, max_iterations + 1):
        iteration_start_hlgs = current_hlgs.model_copy(deep=True)
        iteration_start_llgs = current_llgs.model_copy(deep=True)
        print(
            f"[bottom-up] iteration {iteration}/{max_iterations} starting",
            flush=True,
        )
        branches = group_low_level_goals(current_hlgs, current_llgs)
        iteration_warnings: list[str] = []

        empty_branches = [
            branch for branch in branches if not branch.low_level_goals
        ]
        if empty_branches:
            empty_names = {
                _stable_hlg_key(branch.high_level_goal)
                for branch in empty_branches
            }
            current_hlgs = HighLevelGoals(
                goals=[
                    goal
                    for goal in current_hlgs.goals
                    if _stable_hlg_key(goal) not in empty_names
                ]
            )
            branches = [
                branch
                for branch in branches
                if branch.low_level_goals
            ]
            for branch in empty_branches:
                iteration_warnings.append(
                    f"{branch.branch_id}: EMPTY_BRANCH_REMOVED "
                    "(HLG had no Low-Level Goals)"
                )

        # Apply the first step of the documented cycle before evaluating the
        # branches, so duplicate HLGs do not create parallel branches.
        current_hlgs, current_llgs = _deduplicate_high_level_goals(
            current_hlgs, current_llgs, iteration_warnings
        )
        branches = group_low_level_goals(current_hlgs, current_llgs)

        # A confirmed branch is skipped while the HLG structure is unchanged.
        # Structural changes clear this set after the iteration and trigger a
        # fresh audit. Stable actor/name identity survives branch renumbering.
        active_branches = [
            branch
            for branch in branches
            if _stable_hlg_key(branch.high_level_goal)
            not in confirmed_branch_keys
        ]

        reconstructions = reconstruct_all_branches(active_branches)
        raw_evaluations = evaluate_all_branches(
            project_description,
            active_branches,
            reconstructions,
            current_hlgs,
            stabilized_llg_keys,
            critic_conversation,
        )
        evaluations = dict(raw_evaluations)
        confirmed_branch_keys.update(
            _stable_hlg_key(branch.high_level_goal)
            for branch in active_branches
            if evaluations[branch.branch_id].final_decision
            == GlobalGoalEvaluationDecision.CONFIRM_BRANCH
        )

        known_actors = _merge_actors(
            known_actors, _actors_from_goals(current_hlgs)
        )
        missing_hlg_evaluation = evaluate_missing_high_level_goals(
            project_description,
            current_hlgs,
            known_actors,
            critic_conversation,
        )

        removed_names: set[str] = set()
        llg_regenerations: dict[
            str, tuple[HighLevelGoal, str, list[LowLevelGoal], int]
        ] = {}
        replacements: dict[str, list[HighLevelGoal]] = {}
        additions: list[HighLevelGoal] = []
        reserved_names = {
            *(_stable_hlg_key(goal) for goal in current_hlgs.goals),
            *excluded_hlg_keys,
        }
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
            reserved_names.discard(original_key)
            replacement_goals = []
            # A focused rewrite request yields one candidate at most;
            # accepting every candidate here caused HLG proliferation.
            for goal in generated.goals[:MAX_REPLACEMENT_HLGS_PER_REWRITE]:
                goal_key = _stable_hlg_key(goal)
                if goal_key in reserved_names:
                    continue
                reserved_names.add(goal_key)
                replacement_goals.append(goal)
                llg_regenerations[goal_key] = (
                    goal,
                    request.rationale,
                    list(branch.low_level_goals),
                    len(branch.low_level_goals) + MAX_LLG_GROWTH_PER_REPAIR,
                )
            if replacement_goals:
                hlg_rewrite_counts[original_key] = rewrite_count + 1
                replacements[original_key] = replacement_goals
            else:
                iteration_warnings.append(
                    f"{branch.branch_id}: HLG_REGENERATION_DUPLICATE_IGNORED"
                )
            rewrite_handled = True
            break

        if not rewrite_handled:
            removal_branches = [
                branch for branch in active_branches
                if evaluations[branch.branch_id].final_decision
                == GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
            ]
            for branch in removal_branches[:MAX_HLG_REMOVALS_PER_ITERATION]:
                original_key = _stable_hlg_key(branch.high_level_goal)
                removed_names.add(original_key)
                excluded_hlg_keys.add(original_key)
            if len(removal_branches) > MAX_HLG_REMOVALS_PER_ITERATION:
                iteration_warnings.append(
                    "HLG_REMOVAL_BATCH_LIMIT_REACHED "
                    f"({len(removal_branches)} requested, "
                    f"{MAX_HLG_REMOVALS_PER_ITERATION} applied)"
                )

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
                    stabilized_llg_keys.add(stable_key)
                    evaluations[branch.branch_id] = _with_decision(
                        evaluation,
                        GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED,
                        "LLG regeneration limit reached; keeping the latest LLGs.",
                    )
                    iteration_warnings.append(
                        f"{branch.branch_id}: LLG_REGENERATION_LIMIT_REACHED"
                    )
                    continue
                llg_regeneration_counts[stable_key] = count + 1
                if count + 1 >= MAX_LLG_REGENERATIONS_PER_BRANCH:
                    stabilized_llg_keys.add(stable_key)
                llg_regenerations[stable_key] = (
                    branch.high_level_goal,
                    evaluation.low_level_evaluation.regeneration_feedback
                    or evaluation.rationale,
                    list(branch.low_level_goals),
                    len(branch.low_level_goals) + MAX_LLG_GROWTH_PER_REPAIR,
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
                continue
            next_goals.extend(replacements.get(key, [goal]))
        known_names = {_stable_hlg_key(goal) for goal in next_goals}
        for goal in additions:
            if _stable_hlg_key(goal) not in known_names:
                next_goals.append(goal)
                known_names.add(_stable_hlg_key(goal))

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
            maxima = {
                _stable_hlg_key(item[0]): item[3]
                for item in llg_regenerations.values()
            }
            generated_llgs = _regenerate_llgs_with_top_down(
                LowLevelGoalRegenerationRequest(
                    high_level_goals=HighLevelGoals(goals=target_goals),
                    guidance_by_parent_name=guidance,
                    existing_low_level_goals=LowLevelGoals(
                        low_level_goals=existing_llgs
                    ),
                    max_goals_by_parent_name=maxima,
                ),
                mode,
                evaluator_ablation,
            )
            parents = {_stable_hlg_key(goal): goal for goal in target_goals}
            generated_by_parent: dict[str, list[LowLevelGoal]] = {
                key: [] for key in parents
            }
            for llg in generated_llgs.low_level_goals:
                parent = _resolve_generated_parent(
                    llg.high_level_associated, parents
                )
                if parent is None:
                    # A generator can occasionally return a stale or
                    # paraphrased parent reference.  Do not abort the whole
                    # dataset: discard only this malformed LLG and let the
                    # remaining branches continue through the cycle.
                    iteration_warnings.append(
                        f"LLG '{llg.name}' refers to an unexpected HLG; skipped."
                    )
                    continue
                generated_by_parent[_stable_hlg_key(parent)].append(llg)

            for parent_key, item in llg_regenerations.items():
                parent, _, existing, maximum = item
                accepted, rejection = _accept_regenerated_llgs(
                    parent=parent,
                    existing=existing,
                    generated=generated_by_parent.get(parent_key, []),
                    maximum=maximum,
                )
                next_llgs.extend(accepted)
                if rejection:
                    iteration_warnings.append(
                        f"{parent.name}: {rejection}"
                    )

        current_hlgs = HighLevelGoals(goals=next_goals)
        current_llgs = LowLevelGoals(low_level_goals=next_llgs)

        current_hlgs, current_llgs = _deduplicate_high_level_goals(
            current_hlgs, current_llgs, iteration_warnings
        )

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
            reserved_names = {
                *(_stable_hlg_key(goal) for goal in current_hlgs.goals),
                *excluded_hlg_keys,
            }
            for request_index, request in enumerate(
                missing_hlg_evaluation.missing_goal_requests[
                    :MAX_NEW_HLGS_PER_ITERATION
                ],
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
                    goal_key = _stable_hlg_key(goal)
                    if not goal.name.strip() or goal_key in reserved_names:
                        continue
                    reserved_names.add(goal_key)
                    new_goals.append(goal)
                    additions.append(goal)
                if len(additions) >= MAX_NEW_HLGS_PER_ITERATION:
                    break

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
                        max_goals_by_parent_name={
                            _stable_hlg_key(goal): MAX_LLGS_FOR_NEW_HLG
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
                    parent = _resolve_generated_parent(
                        llg.high_level_associated, parents
                    )
                    if parent is not None:
                        generated_by_parent[_stable_hlg_key(parent)].append(llg)
                valid_additions = []
                for parent in additions:
                    accepted, rejection = _accept_regenerated_llgs(
                        parent=parent,
                        existing=[],
                        generated=generated_by_parent.get(_stable_hlg_key(parent), []),
                        maximum=MAX_LLGS_FOR_NEW_HLG,
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

        current_hlgs, current_llgs = _deduplicate_high_level_goals(
            current_hlgs, current_llgs, iteration_warnings
        )

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

        # Only real confirmations permit convergence. Branches that exhausted
        # their repair budget and inconclusive evaluations remain unresolved;
        # their warnings prevent a false confirmed stop.
        all_confirmed = all(
            evaluation.final_decision
            in {
                GlobalGoalEvaluationDecision.CONFIRM_BRANCH,
            }
            for evaluation in evaluations.values()
        ) and not any(
            "LLG_REGENERATION_LIMIT_REACHED" not in warning
            for warning in iteration_warnings
        )
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
                else GlobalGoalEvaluationDecision.NO_MISSING_HIGH_LEVEL_GOALS
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
