"""
Pure deterministic state-update helpers for the bottom-up feedback loop.

This module never calls an LLM and never invokes HLG/LLG generators. The outer
orchestrator owns all decisions, callback execution, retries, and ordering.

Under the current feedback policy a non-confirmed branch is regenerated from
HLG level: the evaluator requests one replacement HLG, the orchestrator obtains
that HLG through the ORIGINAL HLG Generator -> HLG Evaluator loop, and then
always regenerates the replacement HLG's LLG decomposition through the ORIGINAL
LLG Generator -> LLG Evaluator loop.

This module only:
- validates already generated HLG objects;
- provides deterministic duplicate guards;
- replaces an existing branch HLG with an already generated replacement;
- appends new HLGs produced by global documentation coverage;
- merges already regenerated LLGs while preserving untouched branches.
"""

from src.data_model import (
    HighLevelGoal,
    HighLevelGoalGenerationAction,
    HighLevelGoalGenerationRequest,
    HighLevelGoalGenerationSource,
    HighLevelGoals,
    LowLevelGoal,
    LowLevelGoals,
)
from src.bottom_up.goal_reconstructor import normalize_goal_name
from src.bottom_up.semantic_similarity import (
    find_semantic_duplicate_high_level_goal,
    is_semantic_duplicate_high_level_goal,
)


def _deduplicate_goals(goals: list[HighLevelGoal]) -> HighLevelGoals:
    """Return a deterministic HLG collection with lexical/semantic duplicates removed.

    This helper does not choose between alternative generated outputs and does
    not call any evaluator. It is used only when the orchestrator has already
    decided which HLGs are candidates for a subsequent state update.
    """
    result: list[HighLevelGoal] = []
    seen_names: set[str] = set()

    for goal in goals:
        normalized_name = normalize_goal_name(goal.name)
        if normalized_name in seen_names:
            continue
        if is_semantic_duplicate_high_level_goal(goal, result):
            continue
        seen_names.add(normalized_name)
        result.append(goal)

    return HighLevelGoals(goals=result)


def _validate_generated_high_level_goals(
    request: HighLevelGoalGenerationRequest,
    generated: HighLevelGoals,
) -> tuple[list[HighLevelGoal] | None, str | None]:
    """Validate an HLG result already produced by the original HLG loop.

    The orchestrator calls the ORIGINAL HLG Generator -> HLG Evaluator loop.
    This helper only validates the returned object before it is admitted to the
    state-update path. A focused correction request must produce exactly one
    non-empty HLG for one of the actors supplied in the request.
    """
    if not isinstance(generated, HighLevelGoals):
        return None, (
            "TypeError: evaluated HLG callback must return a HighLevelGoals "
            "instance."
        )

    if len(generated.goals) != 1:
        return None, (
            f"ValueError: focused request '{request.request_id}' must return "
            "exactly one HLG after the original HLG evaluator loop; "
            f"received {len(generated.goals)}."
        )

    allowed_actor_names = {
        normalize_goal_name(actor.name)
        for actor in request.generator_input.actors.actors
    }
    generated_goal = generated.goals[0]

    if not generated_goal.name.strip() or not generated_goal.description.strip():
        return None, (
            f"ValueError: request '{request.request_id}' returned an HLG "
            "with an empty name or description."
        )

    returned_actor_name = normalize_goal_name(generated_goal.actor.name)
    if returned_actor_name not in allowed_actor_names:
        return None, (
            f"ValueError: request '{request.request_id}' returned actor "
            f"'{generated_goal.actor.name}', which was not supplied to the "
            "original HLG generator."
        )

    return [generated_goal], None


def _goals_are_duplicates(
    left_goal: HighLevelGoal,
    right_goal: HighLevelGoal,
) -> bool:
    """Return whether two HLGs represent the same duplicate relation."""
    if normalize_goal_name(left_goal.name) == normalize_goal_name(
        right_goal.name
    ):
        return True

    matched, _ = find_semantic_duplicate_high_level_goal(
        left_goal,
        [right_goal],
    )
    return matched is not None


def _flatten_generated_high_level_goals(
    generated_by_request: dict[str, list[HighLevelGoal]],
) -> list[HighLevelGoal]:
    """Flatten surviving generated HLGs while preserving request/output order."""
    return [
        goal
        for goals in generated_by_request.values()
        for goal in goals
    ]


def _append_unique_high_level_goal(
    goal: HighLevelGoal,
    updated: list[HighLevelGoal],
    seen_names: set[str],
) -> None:
    """Append one HLG to an updated state, failing closed on duplicates."""
    normalized_name = normalize_goal_name(goal.name)
    if normalized_name in seen_names:
        raise ValueError(
            f"HLG update produced duplicate goal name '{goal.name}'."
        )

    matched_existing_goal, similarity = find_semantic_duplicate_high_level_goal(
        goal,
        updated,
    )
    if matched_existing_goal is not None:
        score_details = (
            f" (similarity={similarity:.4f})"
            if similarity is not None
            else ""
        )
        raise ValueError(
            f"HLG update would introduce semantic duplicate '{goal.name}' "
            f"of '{matched_existing_goal.name}'{score_details}."
        )

    seen_names.add(normalized_name)
    updated.append(goal)


def _apply_branch_high_level_generation(
    existing_high_level_goals: dict[str, HighLevelGoal],
    requests: list[HighLevelGoalGenerationRequest],
    generated_by_request: dict[str, list[HighLevelGoal]],
) -> HighLevelGoals:
    """Apply already-generated replacement HLGs for unstable branches.

    Branch evaluation is allowed to do only one structural HLG operation:
    replace the current branch after REGENERATE_HIGH_LEVEL_GOAL. ADD operations
    belong exclusively to documentation coverage and are applied by
    ``_append_coverage_generated_goals``.
    """
    replacements: dict[str, HighLevelGoal] = {}

    for request in requests:
        if request.source != HighLevelGoalGenerationSource.BRANCH_EVALUATION:
            raise ValueError(
                f"Branch request '{request.request_id}' has invalid source "
                f"'{request.source.value}'."
            )
        if (
            request.action
            != HighLevelGoalGenerationAction.REPLACE_EXISTING_HIGH_LEVEL_GOAL
        ):
            raise ValueError(
                "Branch-level feedback may only replace an existing HLG; "
                f"request '{request.request_id}' uses '{request.action.value}'."
            )

        target_branch_id = request.target_branch_id
        if target_branch_id is None or target_branch_id not in existing_high_level_goals:
            raise ValueError(
                f"Request '{request.request_id}' references invalid target "
                f"branch '{target_branch_id}'."
            )
        if request.origin_branch_id != target_branch_id:
            raise ValueError(
                f"Request '{request.request_id}' must originate from and replace "
                f"the same branch '{target_branch_id}'."
            )
        if target_branch_id in replacements:
            raise ValueError(
                f"Multiple replacement requests target branch '{target_branch_id}'."
            )
        if request.request_id not in generated_by_request:
            raise ValueError(
                f"No generation result exists for request '{request.request_id}'."
            )

        generated_goals = generated_by_request[request.request_id]
        if len(generated_goals) != 1:
            raise ValueError(
                f"Replacement request '{request.request_id}' must have exactly "
                f"one surviving evaluator-approved HLG; received "
                f"{len(generated_goals)}."
            )
        replacements[target_branch_id] = generated_goals[0]

    updated: list[HighLevelGoal] = []
    seen_names: set[str] = set()
    for branch_id, original in existing_high_level_goals.items():
        goal_to_keep = replacements.get(branch_id, original)
        _append_unique_high_level_goal(goal_to_keep, updated, seen_names)

    return HighLevelGoals(goals=updated)


def _append_coverage_generated_goals(
    current_high_level_goals: HighLevelGoals,
    requests: list[HighLevelGoalGenerationRequest],
    generated_by_request: dict[str, list[HighLevelGoal]],
) -> tuple[HighLevelGoals, list[HighLevelGoal]]:
    """Append surviving documentation-coverage HLGs to the current HLG state.

    Generation and duplicate-policy decisions have already been performed by
    the orchestrator. This helper only validates the coverage request metadata
    and applies the surviving additions, with a final deterministic duplicate
    guard before the new HLG state is returned.
    """
    current_goals = list(current_high_level_goals.goals)
    seen_names = {
        normalize_goal_name(goal.name)
        for goal in current_goals
    }
    added: list[HighLevelGoal] = []

    for request in requests:
        if (
            request.source
            != HighLevelGoalGenerationSource.DOCUMENTATION_COVERAGE
        ):
            raise ValueError(
                f"Coverage request '{request.request_id}' has invalid source "
                f"'{request.source.value}'."
            )

        if (
            request.action
            != HighLevelGoalGenerationAction.ADD_NEW_HIGH_LEVEL_GOAL
        ):
            raise ValueError("Documentation coverage may only add new HLGs.")

        if request.request_id not in generated_by_request:
            raise ValueError(
                f"No generation result exists for coverage request "
                f"'{request.request_id}'."
            )

        generated_goals = generated_by_request[request.request_id]
        if len(generated_goals) > 1:
            raise ValueError(
                f"Coverage request '{request.request_id}' cannot apply more "
                f"than one focused HLG; received {len(generated_goals)}."
            )

        for generated_goal in generated_goals:
            _append_unique_high_level_goal(
                generated_goal,
                current_goals,
                seen_names,
            )
            added.append(generated_goal)

    return HighLevelGoals(goals=current_goals), added


def _merge_selectively_regenerated_low_level_goals(
    current_low_level_goals: LowLevelGoals,
    regenerated_low_level_goals: LowLevelGoals,
    replaced_parent_names: set[str],
) -> LowLevelGoals:
    """Apply an already regenerated LLG subset to the current LLG state.

    ``replaced_parent_names`` contains normalized HLG names whose previous LLG
    decomposition must be removed. All other LLGs are preserved unchanged.
    The regenerated LLGs have already passed through the ORIGINAL LLG Generator
    -> LLG Evaluator loop before the orchestrator calls this function.
    """
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
