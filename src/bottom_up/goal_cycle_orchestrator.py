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


DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS = 5
MAX_LLG_REGENERATIONS_PER_BRANCH = 2
MAX_NEW_HLGS_PER_ITERATION = 1


def _key(value: str) -> str:
    return " ".join(value.casefold().split())


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


def _actor_for_request(request_actor: str, actors: Actors) -> Actor:
    wanted = _key(request_actor)
    for actor in actors.actors:
        if _key(actor.name) == wanted:
            return actor
    return Actor(
        name=request_actor,
        description="Actor identified in the project documentation.",
    )


def _deduplicate_high_level_goals(
    goals: HighLevelGoals,
    llgs: LowLevelGoals,
    warnings: list[str],
) -> tuple[HighLevelGoals, LowLevelGoals]:
    """Remove conservative name/actor duplicates after one state change."""
    kept: list[HighLevelGoal] = []
    aliases: dict[str, HighLevelGoal] = {}
    for goal in goals.goals:
        key = _key(goal.name)
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
        parent = aliases.get(_key(llg.high_level_associated.name))
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
    )
    if not isinstance(result, HighLevelGoals) or not result.goals:
        raise ValueError("The top-down pipeline generated no High-Level Goals.")
    return result


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
    focused_description = (
        "Evaluator rationale (use this as the primary correction guidance):\n"
        f"{feedback or 'No additional rationale provided.'}\n\n"
        "Generate only the decomposition of these documented stakeholder "
        f"intentions:\n{scope}"
    )

    result, _, _ = generate_response_with_reflection(
        target_type="Low Level Goals",
        call_function=top_down_generate_low_level_goals,
        define_args=(request.high_level_goals,),
        eval_mode=EvalMode.LOW_LEVEL,
        eval_args=(focused_description, actors, request.high_level_goals),
        shotPromptingMode=mode,
        llama_ablation=evaluator_ablation,
    )
    if not isinstance(result, LowLevelGoals):
        raise TypeError("The top-down pipeline did not return LowLevelGoals.")
    return result


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
    llg_regeneration_counts: dict[str, int] = {}
    traces: list[GlobalGoalCycleIteration] = []
    warnings: list[str] = []
    known_actors = _actors_from_goals(initial_high_level_goals)

    for iteration in range(1, max_iterations + 1):
        branches = group_low_level_goals(current_hlgs, current_llgs)
        iteration_warnings: list[str] = []

        empty_branches = [
            branch for branch in branches if not branch.low_level_goals
        ]
        if empty_branches:
            empty_names = {
                _key(branch.high_level_goal.name)
                for branch in empty_branches
            }
            current_hlgs = HighLevelGoals(
                goals=[
                    goal
                    for goal in current_hlgs.goals
                    if _key(goal.name) not in empty_names
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

        reconstructions = reconstruct_all_branches(branches)
        raw_evaluations = evaluate_all_branches(
            project_description, branches, reconstructions, current_hlgs
        )
        evaluations = dict(raw_evaluations)

        removed_names: set[str] = set()
        llg_regenerations: dict[str, tuple[HighLevelGoal, str]] = {}
        replacements: dict[str, list[HighLevelGoal]] = {}
        additions: list[HighLevelGoal] = []
        reserved_names = {_key(goal.name) for goal in current_hlgs.goals}
        # Deterministic action order: rewrite, remove, then LLG regeneration.
        # Only the first applicable action is applied in this iteration.
        priority = {
            GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL: 0,
            GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL: 1,
            GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS: 2,
        }
        ordered_branches = sorted(
            branches,
            key=lambda branch: priority.get(
                evaluations[branch.branch_id].final_decision, 99
            ),
        )
        for branch in ordered_branches:
            evaluation = evaluations[branch.branch_id]
            decision = evaluation.final_decision

            if decision == GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS:
                count = llg_regeneration_counts.get(branch.branch_id, 0)
                if count >= MAX_LLG_REGENERATIONS_PER_BRANCH:
                    evaluations[branch.branch_id] = _with_decision(
                        evaluation,
                        GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED,
                        "LLG regeneration limit reached; keeping the latest LLGs.",
                    )
                    iteration_warnings.append(
                        f"{branch.branch_id}: LLG_REGENERATION_LIMIT_REACHED"
                    )
                else:
                    llg_regeneration_counts[branch.branch_id] = count + 1
                    llg_regenerations[_key(branch.high_level_goal.name)] = (
                        branch.high_level_goal,
                        evaluation.low_level_evaluation.regeneration_feedback
                        or evaluation.rationale,
                    )
                    break

            elif decision == GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL:
                removed_names.add(_key(branch.high_level_goal.name))
                break

            elif decision == GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL:
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
                original_key = _key(branch.high_level_goal.name)
                reserved_names.discard(original_key)
                replacement_goals = []
                # A focused rewrite request yields one candidate at most;
                # accepting every candidate here caused HLG proliferation.
                for goal in generated.goals[:MAX_NEW_HLGS_PER_ITERATION]:
                    goal_key = _key(goal.name)
                    if goal_key in reserved_names:
                        continue
                    reserved_names.add(goal_key)
                    replacement_goals.append(goal)
                    llg_regenerations[goal_key] = (goal, request.rationale)
                if replacement_goals:
                    replacements[original_key] = replacement_goals
                else:
                    # A top-down retry may return only the unchanged HLG. Keep
                    # the current goal and continue evaluating the project.
                    iteration_warnings.append(
                        f"{branch.branch_id}: HLG_REGENERATION_DUPLICATE_IGNORED"
                    )
                break

        # Apply branch-level HLG changes before the one global coverage check.
        next_goals: list[HighLevelGoal] = []
        for goal in current_hlgs.goals:
            key = _key(goal.name)
            if key in removed_names:
                continue
            next_goals.extend(replacements.get(key, [goal]))
        known_names = {_key(goal.name) for goal in next_goals}
        for goal in additions:
            if _key(goal.name) not in known_names:
                next_goals.append(goal)
                known_names.add(_key(goal.name))

        changed_parent_names = removed_names | set(replacements) | set(llg_regenerations)
        next_llgs = [
            llg
            for llg in current_llgs.low_level_goals
            if _key(llg.high_level_associated.name) not in changed_parent_names
        ]

        # The branch evaluator can request LLG repair, but the global HLG
        # coverage check is deliberately a separate, single project-wide call.
        if llg_regenerations:
            target_goals = [goal for goal, _ in llg_regenerations.values()]
            guidance = {
                goal.name: feedback
                for goal, feedback in llg_regenerations.values()
            }
            generated_llgs = _regenerate_llgs_with_top_down(
                LowLevelGoalRegenerationRequest(
                    high_level_goals=HighLevelGoals(goals=target_goals),
                    guidance_by_parent_name=guidance,
                ),
                mode,
                evaluator_ablation,
            )
            parents = {_key(goal.name): goal for goal in target_goals}
            for llg in generated_llgs.low_level_goals:
                parent = parents.get(_key(llg.high_level_associated.name))
                if parent is None:
                    raise ValueError(
                        f"Generated LLG '{llg.name}' refers to an unexpected HLG."
                    )
                next_llgs.append(_replace_parent(llg, parent))

        current_hlgs = HighLevelGoals(goals=next_goals)
        current_llgs = LowLevelGoals(low_level_goals=next_llgs)

        current_hlgs, current_llgs = _deduplicate_high_level_goals(
            current_hlgs, current_llgs, iteration_warnings
        )

        known_actors = _merge_actors(known_actors, _actors_from_goals(current_hlgs))
        actors = known_actors
        branches_ready = all(
            evaluation.final_decision
            in {
                GlobalGoalEvaluationDecision.CONFIRM_BRANCH,
                GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED,
            }
            for evaluation in evaluations.values()
        )
        if branches_ready:
            missing_hlg_evaluation = evaluate_missing_high_level_goals(
                project_description, current_hlgs, actors
            )
        else:
            missing_hlg_evaluation = MissingHighLevelGoalEvaluation(
                decision=MissingHighLevelGoalDecision.NO_MISSING,
                rationale=(
                    "Global coverage check deferred until all current branches "
                    "are confirmed or stabilized."
                ),
            )

        if (
            not llg_regenerations
            and not replacements
            and not removed_names
            and missing_hlg_evaluation.decision == MissingHighLevelGoalDecision.FOUND
        ):
            reserved_names = {_key(goal.name) for goal in current_hlgs.goals}
            for request_index, request in enumerate(
                missing_hlg_evaluation.missing_goal_requests[:1], start=1
            ):
                actor = _actor_for_request(request.actor, actors)
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
                    goal_key = _key(goal.name)
                    if not goal.name.strip() or goal_key in reserved_names:
                        continue
                    reserved_names.add(goal_key)
                    new_goals.append(goal)
                    additions.append(goal)
                    llg_regenerations[goal_key] = (
                        goal,
                        missing_hlg_evaluation.rationale,
                    )
                break

            if additions:
                current_hlgs = HighLevelGoals(
                    goals=[*current_hlgs.goals, *additions]
                )
                generated_llgs = _regenerate_llgs_with_top_down(
                    LowLevelGoalRegenerationRequest(
                        high_level_goals=HighLevelGoals(goals=additions),
                        guidance_by_parent_name={
                            goal.name: missing_hlg_evaluation.rationale
                            for goal in additions
                        },
                    ),
                    mode,
                    evaluator_ablation,
                )
                parents = {_key(goal.name): goal for goal in additions}
                for llg in generated_llgs.low_level_goals:
                    parent = parents.get(_key(llg.high_level_associated.name))
                    if parent is not None:
                        current_llgs.low_level_goals.append(
                            _replace_parent(llg, parent)
                        )

        current_hlgs, current_llgs = _deduplicate_high_level_goals(
            current_hlgs, current_llgs, iteration_warnings
        )

        # A branch that exhausted its bounded LLG repairs is considered
        # stabilized for the final project-wide coverage check.  Its warning
        # is preserved, but it must not prevent convergence when no HLG is
        # missing. Other warnings (for example an empty branch removed) still
        # block the normal confirmed stop.
        all_confirmed = all(
            evaluation.final_decision
            in {
                GlobalGoalEvaluationDecision.CONFIRM_BRANCH,
                GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED,
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
