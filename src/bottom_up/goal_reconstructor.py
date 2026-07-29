"""
goal_reconstructor.py

Bottom-up High-Level Goal Generator.

Given the low-level goals produced (top-down) for a given high-level goal branch,
this module reconstructs a candidate high-level goal that explains the shared
intention behind that group of low-level goals ("why would this set of low-level
goals be implemented?"), without ever seeing the original high-level goal that
produced them.

Branch grouping, identifier assignment and parent traceability are handled
entirely by Python. Only branch-local, opaque identifiers and the low-level
goals' own content (plus, optionally, the associated actor) are ever included
in the LLM prompt.
"""

from src.data_model import (
    Actor,
    HighLevelGoal,
    HighLevelGoals,
    LowLevelGoal,
    LowLevelGoals,
    BottomUpHighLevelGoal,
    BottomUpHighLevelGoalLLMOutput,
)
from src.llm_clients import generate_response


class BranchAssignmentError(ValueError):
    """
    Raised when one or more low-level goals cannot be deterministically
    assigned to a branch, or when the grouped total does not match the
    original number of low-level goals. Unassigned goals must never be
    silently dropped.
    """


def normalize_goal_name(name: str) -> str:
    """
    Normalizes a goal name before branch matching.

    The normalization:
    - removes leading and trailing whitespace;
    - replaces consecutive whitespace with a single space;
    - performs case-insensitive normalization through casefold().

    This avoids branch-assignment failures caused only by irrelevant
    differences in capitalization or spacing.
    """
    return " ".join(name.casefold().strip().split())


def assign_branch_ids(
    high_level_goals: HighLevelGoals,
) -> dict[str, HighLevelGoal]:
    """
    Assigns an opaque, deterministic branch_id
    ('branch_001', 'branch_002', ...) to each high-level goal, in the
    order they appear in high_level_goals.goals.

    These ids carry no semantic content from the original high-level goal,
    so that the bottom-up generator cannot be anchored by the parent's
    wording.
    """
    return {
        f"branch_{i + 1:03d}": hlg
        for i, hlg in enumerate(high_level_goals.goals)
    }


def group_low_level_goals_by_branch(
    low_level_goals: LowLevelGoals,
    branch_map: dict[str, HighLevelGoal],
) -> dict[str, list[LowLevelGoal]]:
    """
    Groups low-level goals by branch_id, using the embedded parent
    (LowLevelGoal.high_level_associated.name) purely as a Python-side join
    key against branch_map.

    Goal names are normalized before matching. The original parent name
    and the branch identifier are never exposed to the LLM.

    Every branch_id in branch_map is present in the result, even if empty.

    Raises BranchAssignmentError if:
    - two or more high-level goals have the same normalized name;
    - a low-level goal cannot be matched to a branch;
    - the grouped total does not match the input total.
    """
    normalized_names: dict[str, list[str]] = {}

    for branch_id, high_level_goal in branch_map.items():
        normalized_name = normalize_goal_name(high_level_goal.name)
        normalized_names.setdefault(normalized_name, []).append(branch_id)

    duplicates = {
        normalized_name: branch_ids
        for normalized_name, branch_ids in normalized_names.items()
        if len(branch_ids) > 1
    }

    if duplicates:
        duplicate_details = "; ".join(
            f"'{name}' -> {branch_ids}"
            for name, branch_ids in duplicates.items()
        )

        raise BranchAssignmentError(
            "Duplicate high-level goal names prevent deterministic "
            f"branch assignment after normalization: {duplicate_details}"
        )

    name_to_branch_id = {
        normalized_name: branch_ids[0]
        for normalized_name, branch_ids in normalized_names.items()
    }

    branches: dict[str, list[LowLevelGoal]] = {
        branch_id: []
        for branch_id in branch_map
    }

    unassigned: list[LowLevelGoal] = []

    for low_level_goal in low_level_goals.low_level_goals:
        normalized_parent_name = normalize_goal_name(
            low_level_goal.high_level_associated.name
        )

        branch_id = name_to_branch_id.get(normalized_parent_name)

        if branch_id is None:
            unassigned.append(low_level_goal)
        else:
            branches[branch_id].append(low_level_goal)

    if unassigned:
        details = "; ".join(
            (
                f"'{goal.name}' "
                f"(parent: '{goal.high_level_associated.name}')"
            )
            for goal in unassigned
        )

        raise BranchAssignmentError(
            f"{len(unassigned)} low-level goal(s) could not be assigned "
            f"to any branch: {details}"
        )

    total_grouped = sum(
        len(goals)
        for goals in branches.values()
    )

    total_input = len(low_level_goals.low_level_goals)

    if total_grouped != total_input:
        raise BranchAssignmentError(
            "Mismatch between the number of grouped low-level goals "
            f"({total_grouped}) and the total number of input "
            f"low-level goals ({total_input})."
        )

    return branches


def assign_local_goal_ids(
    low_level_goals: list[LowLevelGoal],
) -> dict[str, LowLevelGoal]:
    """
    Assigns opaque, deterministic local ids
    ('llg_001', 'llg_002', ...) to the low-level goals of a single branch,
    in list order.

    These ids are scoped to a single prompt and deliberately do not embed
    the branch_id, so that the branch identifier never appears in the LLM
    prompt.
    """
    return {
        f"llg_{i + 1:03d}": goal
        for i, goal in enumerate(low_level_goals)
    }


def reconstruct_high_level_goal(
    branch_id: str,
    low_level_goals: list[LowLevelGoal],
    actor: Actor | None = None,
) -> BottomUpHighLevelGoal:
    """
    Calls the LLM once to reconstruct a candidate high-level goal for a
    single branch.

    The function receives:
    - branch_id, used only for Python-side traceability;
    - the low-level goals belonging to the branch;
    - optionally, the actor already associated with the branch.

    The branch_id is never inserted into the prompt. No content from the
    original parent high-level goal is made available to the LLM.
    """
    if not low_level_goals:
        raise ValueError(
            f"Cannot reconstruct a high-level goal for branch "
            f"'{branch_id}': no low-level goals were provided."
        )

    local_ids = assign_local_goal_ids(low_level_goals)

    sys_prompt = (
        "You are a helpful assistant expert in software engineering tasks, "
        "specialised in the Goal-Oriented Requirements Engineering (GORE) "
        "framework.\n\n"
        "You are given a set of low-level goals that belong to the same "
        "branch of a goal hierarchy. Low-level goals are concrete and "
        "operational objectives that describe how something is achieved.\n\n"
        "Your task is to infer, in a bottom-up way, the single high-level "
        "goal that best explains WHY this specific set of low-level goals "
        "would be pursued together.\n\n"
        "A high-level goal expresses an abstract stakeholder or system "
        "intention and is independent of the concrete operational steps "
        "used to achieve it. It must not be a summary or enumeration of "
        "the low-level actions.\n\n"
        "Rules:\n"
        "- Produce exactly one candidate high-level goal.\n"
        "- Do NOT simply concatenate, enumerate, summarise, or paraphrase "
        "the low-level goals.\n"
        "- Infer the shared underlying intention that justifies the whole "
        "group.\n"
        "- The reconstructed goal must be more abstract than each "
        "individual low-level goal.\n"
        "- Use only the provided low-level goals as evidence.\n"
        "- Do not introduce functions, actors, constraints, technologies, "
        "domain facts, quality attributes, or objectives that are not "
        "supported by the provided low-level goals.\n"
        "- If an actor is provided, you may reference it, but do not invent "
        "a new actor.\n"
        "- If the low-level goals appear heterogeneous or unrelated, still "
        "produce the most plausible reconstructed goal, but explicitly "
        "reflect this uncertainty by assigning low values to 'cohesion' "
        "and 'confidence'.\n"
        "- In 'supporting_low_level_goal_ids', reference ONLY the goal "
        "identifiers given to you below, for example 'llg_001'. Do not "
        "invent new identifiers.\n"
        "- Do not evaluate whether the reconstructed goal is correct with "
        "respect to the project documentation, the original parent goal, "
        "or other high-level goals.\n"
        "- Generate ONLY functional goals.\n"
    )

    goals_block = "\n".join(
        f"- [{local_id}] {goal.name}: {goal.description}"
        for local_id, goal in local_ids.items()
    )

    actor_block = (
        f"**Actor:** {actor.name} - {actor.description}\n\n"
        if actor is not None
        else ""
    )

    prompt = (
        f"{actor_block}"
        "**Low-level goals in this branch:**\n"
        f"{goals_block}\n\n"
        "Based only on the low-level goals listed above, reconstruct the "
        "single functional high-level goal that explains why this specific "
        "set of low-level goals would be implemented.\n\n"
        "**Output:**"
    )

    llm_output: BottomUpHighLevelGoalLLMOutput = generate_response(
        prompt,
        sys_prompt,
        BottomUpHighLevelGoalLLMOutput,
    )

    supporting_ids = llm_output.supporting_low_level_goal_ids

    if not supporting_ids:
        raise ValueError(
            f"Branch '{branch_id}': the model returned no supporting "
            "low-level goal identifiers."
        )

    if any(
        not isinstance(local_id, str) or not local_id.strip()
        for local_id in supporting_ids
    ):
        raise ValueError(
            f"Branch '{branch_id}': the model returned one or more empty "
            "or invalid supporting low-level goal identifiers."
        )

    if len(supporting_ids) != len(set(supporting_ids)):
        raise ValueError(
            f"Branch '{branch_id}': duplicate supporting low-level goal "
            "identifiers were returned."
        )

    valid_local_ids = set(local_ids.keys())

    invalid_ids = [
        local_id
        for local_id in supporting_ids
        if local_id not in valid_local_ids
    ]

    if invalid_ids:
        raise ValueError(
            f"Branch '{branch_id}': the model referenced unknown "
            f"low-level goal id(s) {invalid_ids}. Valid ids for this "
            f"branch are: {sorted(valid_local_ids)}."
        )

    supporting_full_ids = [
        f"{branch_id}_{local_id}"
        for local_id in supporting_ids
    ]

    source_full_ids = [
        f"{branch_id}_{local_id}"
        for local_id in local_ids
    ]

    return BottomUpHighLevelGoal(
        branch_id=branch_id,
        reconstructed_high_level_goal=(
            llm_output.reconstructed_high_level_goal
        ),
        abstraction_rationale=llm_output.abstraction_rationale,
        supporting_low_level_goal_ids=supporting_full_ids,
        source_low_level_goal_ids=source_full_ids,
        cohesion=llm_output.cohesion,
        confidence=llm_output.confidence,
    )


def build_branch_traceability(
    branch_map: dict[str, HighLevelGoal],
) -> dict[str, dict[str, object]]:
    """
    Builds a serializable structure that associates each opaque branch_id
    with its original high-level goal.

    This structure is intended only for Python-side persistence and for the
    future Global Evaluator. It is never passed to the bottom-up generator
    and is never included in the LLM prompt.
    """
    return {
        branch_id: {
            "original_high_level_goal": parent.model_dump(),
        }
        for branch_id, parent in branch_map.items()
    }


def reconstruct_all_branches(
    high_level_goals: HighLevelGoals,
    low_level_goals: LowLevelGoals,
) -> tuple[
    dict[str, BottomUpHighLevelGoal],
    dict[str, HighLevelGoal],
    dict[str, str],
    list[str],
    dict[str, dict[str, object]],
]:
    """
    Orchestration helper that:
    - assigns opaque branch identifiers;
    - groups low-level goals by their original branch;
    - validates complete branch assignment;
    - reconstructs one high-level goal for each non-empty branch;
    - records branch-specific errors without interrupting other branches;
    - records high-level goals with no low-level decomposition;
    - prepares serializable branch traceability for persistence.

    The original parent high-level goal is used only:
    - to construct the branch map;
    - to identify the actor associated with the branch;
    - to create the separate traceability structure.

    The parent high-level goal is never forwarded to
    reconstruct_high_level_goal and is never included in the LLM prompt.

    Returns:
    - results:
      {branch_id: BottomUpHighLevelGoal}

    - branch_map:
      {branch_id: HighLevelGoal}

    - errors:
      {branch_id: error message}

    - empty_branches:
      list of branch_ids whose original high-level goal has no associated
      low-level goals

    - branch_traceability:
      serializable mapping between branch_id and original parent goal,
      intended to be saved in the output JSON
    """
    branch_map = assign_branch_ids(high_level_goals)

    grouped = group_low_level_goals_by_branch(
        low_level_goals,
        branch_map,
    )

    results: dict[str, BottomUpHighLevelGoal] = {}
    errors: dict[str, str] = {}
    empty_branches: list[str] = []

    for branch_id, goals in grouped.items():
        if not goals:
            empty_branches.append(branch_id)
            continue

        parent_high_level_goal = branch_map[branch_id]
        actor = parent_high_level_goal.actor

        try:
            results[branch_id] = reconstruct_high_level_goal(
                branch_id=branch_id,
                low_level_goals=goals,
                actor=actor,
            )
        except Exception as exc:
            errors[branch_id] = (
                f"{type(exc).__name__}: {exc}"
            )

    branch_traceability = build_branch_traceability(branch_map)

    return (
        results,
        branch_map,
        errors,
        empty_branches,
        branch_traceability,
    )