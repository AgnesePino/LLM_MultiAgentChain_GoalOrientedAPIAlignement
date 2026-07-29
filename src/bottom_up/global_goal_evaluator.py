"""
global_goal_evaluator.py

Global Goal Evaluator.

For each branch, compares the bottom-up reconstructed high-level goal against
its original parent, the complete set of already-extracted high-level goals,
and the project description, and decides how the high-level goal collection
should evolve. Low-level goals are never part of its input, are never shown
in its prompt, and are never evaluated directly by this module.

The evaluator only produces decisions and changes to the collection of
high-level goals: it never regenerates low-level goals itself. The pipeline
(orchestrator) is responsible for re-invoking the existing low-level goal
generation stage after the high-level goal collection has been updated.
`regenerate_low_level_goals_if_needed` accepts that existing generator as an
injected callable rather than importing it.
"""
import json
from typing import Callable

from src.data_model import (
    Actor,
    HighLevelGoal,
    HighLevelGoals,
    LowLevelGoals,
    BottomUpHighLevelGoal,
    GlobalGoalEvaluationLLMOutput,
    GlobalGoalEvaluationResult,
)
from src.bottom_up.goal_reconstructor import normalize_goal_name
from src.llm_clients import generate_response_llama


class GlobalGoalEvaluationError(ValueError):
    """
    Raised when the LLM's structured output is inconsistent with its own
    decision (e.g. MATCHES_OTHER_HIGH_LEVEL_GOAL without a resolvable
    matched_high_level_goal_id, or ADD_NEW_HIGH_LEVEL_GOAL/
    REWRITE_ORIGINAL_HIGH_LEVEL_GOAL without a proposed name/description).
    Never silently coerced or guessed.
    """


_OPERATIONAL_ACTION_BY_DECISION: dict[str, str] = {
    "CONFIRM_BRANCH": "KEEP_PARENT",
    "REGENERATE_LOW_LEVEL_GOALS": "KEEP_PARENT_AND_REGENERATE",
    "MATCHES_OTHER_HIGH_LEVEL_GOAL": "KEEP_PARENT_AND_REGENERATE",
    "ADD_NEW_HIGH_LEVEL_GOAL": "ADD_NEW_GOAL_AND_REGENERATE",
    "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL": "REPLACE_PARENT_AND_REGENERATE",
}


def _build_system_prompt() -> str:
    return (
        "You are a helpful assistant, expert in Software Engineering, Requirements "
        "Engineering, and specialised in the Goal-Oriented Requirements Engineering "
        "(GORE) framework.\n\n"

        "You are performing a GLOBAL STRUCTURAL EVALUATION of the high-level goal "
        "hierarchy of a software project. You are given one branch at a time: the "
        "branch's original parent high-level goal, a candidate high-level goal that "
        "was independently reconstructed bottom-up from that branch's low-level "
        "goals, the project description, and the complete set of high-level goals "
        "already extracted for the project.\n\n"

        "Your task is to compare the reconstructed candidate against the original "
        "parent, against every other existing high-level goal, and against the "
        "project description, using SEMANTIC comparison based on shared intention "
        "and meaning, never lexical or string comparison.\n\n"

        "You must return exactly one of the following five decisions:\n"

        "- CONFIRM_BRANCH: the reconstructed candidate is semantically equivalent "
        "to the original parent and fully expresses the same intention. The original "
        "parent is kept unchanged.\n"

        "- REGENERATE_LOW_LEVEL_GOALS: the reconstructed candidate is coherent with "
        "the original parent, but represents only part of its intention. The original "
        "parent is kept unchanged, but its low-level goals must be regenerated.\n"

        "- MATCHES_OTHER_HIGH_LEVEL_GOAL: the reconstructed candidate does not "
        "correspond to its own original parent, but is semantically equivalent to a "
        "DIFFERENT high-level goal already present in the existing set. You must "
        "identify that different goal through its branch_id. The original parent is "
        "kept unchanged.\n"

        "- ADD_NEW_HIGH_LEVEL_GOAL: the reconstructed candidate does not coincide "
        "with the original parent or with any other existing high-level goal, "
        "represents an autonomous functional intention, and is supported by the "
        "project description. A new high-level goal must be added to the collection, "
        "while the original parent is kept unchanged.\n"

        "- REWRITE_ORIGINAL_HIGH_LEVEL_GOAL: use this decision only when the project "
        "description provides positive evidence that the original parent itself is "
        "incorrect, ambiguous, poorly formulated, incoherent, or unsupported. Do not "
        "use this decision merely because the reconstructed candidate differs from "
        "the original parent. The original parent must be replaced with a new "
        "formulation that you propose.\n\n"

        "Apply the following decision order:\n"

        "1. Check whether the reconstructed candidate is fully equivalent to the "
        "original parent.\n"

        "2. Check whether the reconstructed candidate represents only part of the "
        "original parent's intention.\n"

        "3. Check whether the reconstructed candidate matches another existing "
        "high-level goal.\n"

        "4. Check whether the reconstructed candidate represents a new autonomous "
        "functional intention supported by the project description.\n"

        "5. Choose REWRITE_ORIGINAL_HIGH_LEVEL_GOAL only when the project description "
        "positively demonstrates that the original parent itself is incorrect or "
        "unsupported.\n\n"

        "Hard rules:\n"

        "- Never invent functionality, actors, goals, constraints, domain facts, or "
        "system behaviour that are not supported by the project description, the "
        "original parent, or the reconstructed candidate.\n"

        "- Never choose ADD_NEW_HIGH_LEVEL_GOAL without comparing the reconstructed "
        "candidate against every high-level goal in the complete existing set.\n"

        "- You are not given the content or identifiers of the low-level goals. Do "
        "not infer, enumerate, inspect, propose, or evaluate individual low-level "
        "goals.\n"

        "- Use only the reconstructed candidate as evidence of the intention "
        "expressed by the previous low-level decomposition.\n"

        "- Never regenerate, propose, or describe low-level goals yourself. You only "
        "decide whether the existing pipeline must regenerate them.\n"

        "- For MATCHES_OTHER_HIGH_LEVEL_GOAL, matched_high_level_goal_id must contain "
        "the branch_id of a different existing high-level goal.\n"

        "- For ADD_NEW_HIGH_LEVEL_GOAL and REWRITE_ORIGINAL_HIGH_LEVEL_GOAL, both "
        "new_or_replacement_goal_name and new_or_replacement_goal_description must "
        "contain non-empty strings.\n"

        "- For every other decision, new_or_replacement_goal_name and "
        "new_or_replacement_goal_description must be null.\n"

        "- matched_high_level_goal_id must be null for every decision except "
        "MATCHES_OTHER_HIGH_LEVEL_GOAL.\n"

        "- Generate only functional content.\n"

        "- Do not provide chain-of-thought, intermediate reasoning steps, Markdown, "
        "comments, explanations outside the requested fields, or code fences.\n\n"

        "Return exactly one valid JSON object with the following fields:\n"

        '{\n'
        '  "decision": "CONFIRM_BRANCH | REGENERATE_LOW_LEVEL_GOALS | '
        'MATCHES_OTHER_HIGH_LEVEL_GOAL | ADD_NEW_HIGH_LEVEL_GOAL | '
        'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",\n'
        '  "matched_high_level_goal_id": "string or null",\n'
        '  "new_or_replacement_goal_name": "string or null",\n'
        '  "new_or_replacement_goal_description": "string or null",\n'
        '  "rationale": "string"\n'
        '}\n\n'

        "The rationale must briefly justify the selected decision using only the "
        "provided project description, original parent, reconstructed candidate, and "
        "existing high-level goals.\n"

        "Use null for every optional field that is not relevant to the selected "
        "decision.\n"

        "Respond only with the JSON object."
    )


def _build_user_prompt(
    branch_id: str,
    project_description: str,
    original_high_level_goal: HighLevelGoal,
    reconstructed_high_level_goal: BottomUpHighLevelGoal,
    existing_high_level_goals: dict[str, HighLevelGoal],
) -> str:
    existing_block = "\n".join(
        f"- [{existing_branch_id}] {hlg.name}: {hlg.description} "
        f"(actor: {hlg.actor.name})"
        for existing_branch_id, hlg in existing_high_level_goals.items()
    )

    return (
        f"**branch_id:** {branch_id}\n\n"
        "**Project description:**\n"
        f"{project_description}\n\n"
        "**Original parent high-level goal:**\n"
        f"- Actor: {original_high_level_goal.actor.name} - "
        f"{original_high_level_goal.actor.description}\n"
        f"- Name: {original_high_level_goal.name}\n"
        f"- Description: {original_high_level_goal.description}\n\n"
        "**Bottom-up reconstructed candidate for this branch:**\n"
        f"- Reconstructed goal: {reconstructed_high_level_goal.reconstructed_high_level_goal}\n"
        f"- Abstraction rationale: {reconstructed_high_level_goal.abstraction_rationale}\n"
        f"- Cohesion: {reconstructed_high_level_goal.cohesion}\n"
        f"- Confidence: {reconstructed_high_level_goal.confidence}\n\n"
        "**Complete set of existing high-level goals (stable identifiers):**\n"
        f"{existing_block}\n\n"
        "Evaluate this branch and return the structured decision now.\n"
        "**Output:**"
    )


def _resolve_matched_goal(
    branch_id: str,
    matched_high_level_goal_id: str | None,
    existing_high_level_goals: dict[str, HighLevelGoal],
) -> HighLevelGoal:
    if not matched_high_level_goal_id:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': decision MATCHES_OTHER_HIGH_LEVEL_GOAL requires "
            "a matched_high_level_goal_id, but none was returned."
        )

    if matched_high_level_goal_id == branch_id:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': MATCHES_OTHER_HIGH_LEVEL_GOAL cannot reference "
            "the branch's own original parent as the matched goal."
        )

    matched = existing_high_level_goals.get(matched_high_level_goal_id)
    if matched is None:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': the model referenced an unknown "
            f"matched_high_level_goal_id '{matched_high_level_goal_id}'. Valid ids "
            f"are: {sorted(existing_high_level_goals.keys())}."
        )

    return matched


def _build_high_level_goal_from_llm_text(
    branch_id: str,
    decision: str,
    llm_output: GlobalGoalEvaluationLLMOutput,
    actor: Actor,
) -> HighLevelGoal:
    if not llm_output.new_or_replacement_goal_name or not llm_output.new_or_replacement_goal_description:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': decision {decision} requires both "
            "new_or_replacement_goal_name and new_or_replacement_goal_description, "
            "but at least one was missing."
        )

    return HighLevelGoal(
        name=llm_output.new_or_replacement_goal_name,
        description=llm_output.new_or_replacement_goal_description,
        actor=actor,
    )



def _validate_decision_fields(
    branch_id: str,
    llm_output: GlobalGoalEvaluationLLMOutput,
) -> None:
    """Validates that only fields relevant to the selected decision are set."""
    decision = llm_output.decision

    if decision != "MATCHES_OTHER_HIGH_LEVEL_GOAL" and (
        llm_output.matched_high_level_goal_id is not None
    ):
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': matched_high_level_goal_id is only valid for "
            "MATCHES_OTHER_HIGH_LEVEL_GOAL."
        )

    goal_text_decisions = {
        "ADD_NEW_HIGH_LEVEL_GOAL",
        "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",
    }

    if decision not in goal_text_decisions and (
        llm_output.new_or_replacement_goal_name is not None
        or llm_output.new_or_replacement_goal_description is not None
    ):
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': new_or_replacement_goal_name and "
            "new_or_replacement_goal_description are only valid for "
            "ADD_NEW_HIGH_LEVEL_GOAL or REWRITE_ORIGINAL_HIGH_LEVEL_GOAL."
        )

def _parse_llama_evaluation(
    raw_response: str,
) -> GlobalGoalEvaluationLLMOutput:
    """
    Converts the textual JSON response returned by Llama into the
    structured Pydantic model expected by the Global Goal Evaluator.
    """
    if not raw_response or not raw_response.strip():
        raise GlobalGoalEvaluationError(
            "Llama returned an empty Global Goal Evaluation response."
        )

    cleaned_response = raw_response.strip()

    # Remove possible Markdown code fences.
    if cleaned_response.startswith("```"):
        lines = cleaned_response.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned_response = "\n".join(lines).strip()

    try:
        parsed_json = json.loads(cleaned_response)
    except json.JSONDecodeError as exc:
        raise GlobalGoalEvaluationError(
            "Llama did not return valid JSON for the Global Goal Evaluator. "
            f"Response received: {raw_response}"
        ) from exc

    try:
        return GlobalGoalEvaluationLLMOutput.model_validate(parsed_json)
    except Exception as exc:
        raise GlobalGoalEvaluationError(
            "Llama returned JSON that does not respect the expected "
            "GlobalGoalEvaluationLLMOutput structure. "
            f"Response received: {raw_response}"
        ) from exc

def evaluate_branch(
    branch_id: str,
    project_description: str,
    original_high_level_goal: HighLevelGoal,
    reconstructed_high_level_goal: BottomUpHighLevelGoal,
    existing_high_level_goals: dict[str, HighLevelGoal],
) -> GlobalGoalEvaluationResult:
    """
    Evaluates a single branch. Receives only the project description, the
    branch's original parent, its bottom-up reconstructed candidate, and the
    complete set of existing high-level goals (keyed by stable branch_id).
    No low-level goal is ever part of the input, the prompt, or the returned
    result.
    """
    sys_prompt = _build_system_prompt()
    prompt = _build_user_prompt(
        branch_id,
        project_description,
        original_high_level_goal,
        reconstructed_high_level_goal,
        existing_high_level_goals,
    )

    raw_llm_output = generate_response_llama(
        prompt,
        sys_prompt,
    )

    llm_output: GlobalGoalEvaluationLLMOutput = (
        _parse_llama_evaluation(raw_llm_output)
    )

    _validate_decision_fields(branch_id, llm_output)

    decision = llm_output.decision
    actor = original_high_level_goal.actor

    matched_high_level_goal: HighLevelGoal | None = None
    new_or_replacement_high_level_goal: HighLevelGoal | None = None

    if decision == "MATCHES_OTHER_HIGH_LEVEL_GOAL":
        matched_high_level_goal = _resolve_matched_goal(
            branch_id, llm_output.matched_high_level_goal_id, existing_high_level_goals
        )

    elif decision in ("ADD_NEW_HIGH_LEVEL_GOAL", "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL"):
        new_or_replacement_high_level_goal = _build_high_level_goal_from_llm_text(
            branch_id, decision, llm_output, actor
        )

    elif decision not in ("CONFIRM_BRANCH", "REGENERATE_LOW_LEVEL_GOALS"):
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': unknown decision '{decision}' returned by the model."
        )

    return GlobalGoalEvaluationResult(
        branch_id=branch_id,
        decision=decision,
        original_high_level_goal=original_high_level_goal,
        reconstructed_high_level_goal=reconstructed_high_level_goal.reconstructed_high_level_goal,
        matched_high_level_goal=matched_high_level_goal,
        new_or_replacement_high_level_goal=new_or_replacement_high_level_goal,
        rationale=llm_output.rationale,
        operational_action=_OPERATIONAL_ACTION_BY_DECISION[decision],
        requires_low_level_regeneration=decision != "CONFIRM_BRANCH",
    )


def evaluate_all_branches(
    project_description: str,
    existing_high_level_goals: dict[str, HighLevelGoal],
    reconstructed_high_level_goals: dict[str, BottomUpHighLevelGoal],
) -> tuple[dict[str, GlobalGoalEvaluationResult], dict[str, str]]:
    """
    Evaluates every branch for which a bottom-up reconstruction result exists.
    A failure on one branch (invalid/inconsistent LLM output, or any other
    exception) is recorded in the returned errors dict and does not interrupt
    the evaluation of the other branches.
    """
    results: dict[str, GlobalGoalEvaluationResult] = {}
    errors: dict[str, str] = {}

    for branch_id, reconstructed in reconstructed_high_level_goals.items():
        original = existing_high_level_goals.get(branch_id)
        if original is None:
            errors[branch_id] = (
                f"GlobalGoalEvaluationError: branch '{branch_id}' has a bottom-up "
                "result but no corresponding original high-level goal."
            )
            continue

        try:
            results[branch_id] = evaluate_branch(
                branch_id,
                project_description,
                original,
                reconstructed,
                existing_high_level_goals,
            )
        except Exception as exc:
            errors[branch_id] = f"{type(exc).__name__}: {exc}"

    return results, errors


def apply_global_evaluations(
    existing_high_level_goals: dict[str, HighLevelGoal],
    evaluations: dict[str, GlobalGoalEvaluationResult],
) -> HighLevelGoals:
    """
    Pure Python, no LLM call. Builds the updated high-level goal collection
    from the evaluator's decisions.

    Every branch's original parent is kept unless its decision is
    REWRITE_ORIGINAL_HIGH_LEVEL_GOAL (replaced) - this explicitly includes
    ADD_NEW_HIGH_LEVEL_GOAL, whose parent is always kept alongside the newly
    added goal. ADD_NEW_HIGH_LEVEL_GOAL candidates are appended unless a goal
    with the same normalized name (normalize_goal_name) is already present in
    the collection being built - a minimal duplicate check, not advanced
    semantic deduplication.
    """
    updated_goals: list[HighLevelGoal] = []
    seen_normalized_names: set[str] = set()

    for branch_id, original in existing_high_level_goals.items():
        evaluation = evaluations.get(branch_id)

        if evaluation is not None and evaluation.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL":
            resulting = evaluation.new_or_replacement_high_level_goal
            if resulting is None:
                raise GlobalGoalEvaluationError(
                    f"Branch '{branch_id}': missing replacement high-level goal."
                )
        else:
            resulting = original

        updated_goals.append(resulting)
        seen_normalized_names.add(normalize_goal_name(resulting.name))

    for evaluation in evaluations.values():
        if evaluation.decision != "ADD_NEW_HIGH_LEVEL_GOAL":
            continue

        new_goal = evaluation.new_or_replacement_high_level_goal
        if new_goal is None:
            raise GlobalGoalEvaluationError(
                f"Branch '{evaluation.branch_id}': missing new high-level goal."
            )

        normalized_name = normalize_goal_name(new_goal.name)
        if normalized_name in seen_normalized_names:
            continue

        seen_normalized_names.add(normalized_name)
        updated_goals.append(new_goal)

    return HighLevelGoals(goals=updated_goals)


def requires_regeneration(evaluations: dict[str, GlobalGoalEvaluationResult]) -> bool:
    """True iff at least one branch's decision requires low-level goal regeneration."""
    return any(evaluation.requires_low_level_regeneration for evaluation in evaluations.values())


def regenerate_low_level_goals_if_needed(
    evaluations: dict[str, GlobalGoalEvaluationResult],
    updated_high_level_goals: HighLevelGoals,
    regenerate: Callable[[HighLevelGoals], LowLevelGoals],
) -> LowLevelGoals | None:
    """
    Triggers low-level goal regeneration only if at least one evaluation
    requires it. `regenerate` must be the caller's existing low-level goal
    generation pipeline (generate_response_with_reflection(...,
    generate_low_level_goals, ...)) - this function does not implement any
    generation logic itself, it only decides whether to invoke it.
    """
    if not requires_regeneration(evaluations):
        return None

    return regenerate(updated_high_level_goals)


def select_low_level_goals_for_mapping(
    regenerated: LowLevelGoals | None,
    baseline: LowLevelGoals,
) -> LowLevelGoals:
    """Prefers the regenerated low-level goals when available, falling back to the baseline ones."""
    return regenerated if regenerated is not None else baseline
