"""
global_goal_evaluator.py

Global Goal Evaluator.

For each branch, compares the bottom-up reconstructed high-level goal against
its original parent, the complete set of already-extracted high-level goals,
and the project description, and decides how the high-level goal collection
should evolve. Low-level goals are never part of its input, are never shown
in its prompt, and are never evaluated directly by this module.

The evaluator only produces validated decisions and one complete JSON output.
It never applies changes to the high-level goal collection and never regenerates
low-level goals. The JSON is ready for orchestration only when every expected
branch has one valid evaluation and no unresolved error exists. The orchestrator
is solely responsible for loading that JSON, updating goals, and invoking
selective regeneration.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from src.data_model import (
    Actor,
    HighLevelGoal,
    HighLevelGoals,
    BottomUpHighLevelGoal,
    GlobalGoalEvaluationLLMOutput,
    GlobalGoalEvaluationResult,
    DocumentationCoverageLLMOutput,
    DocumentationCoverageResult,
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
    "MATCHES_OTHER_HIGH_LEVEL_GOAL": (
        "REGENERATE_ORIGINAL_BRANCH_AND_RECORD_MISASSIGNMENT"
    ),
    "ADD_NEW_HIGH_LEVEL_GOAL": "ADD_NEW_GOAL_AND_REGENERATE",
    "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL": "REPLACE_PARENT_AND_REGENERATE",
}


def _build_system_prompt() -> str:
    return (
        "You are an expert in Software Engineering, Requirements Engineering, "
        "and Goal-Oriented Requirements Engineering (GORE).\n\n"

        "You must evaluate one high-level goal branch by comparing:\n"
        "- the original parent high-level goal;\n"
        "- the candidate high-level goal reconstructed bottom-up;\n"
        "- the complete set of existing high-level goals;\n"
        "- the complete project description provided in the user prompt.\n\n"

        "The complete project description is the primary source of truth. "
        "Read and consider the entire project description before making any "
        "decision. Do not focus only on isolated keywords, individual sentences, "
        "or the section that appears most similar to the current goal.\n\n"

        "The original parent, the reconstructed candidate, and all other existing "
        "high-level goals were produced by previous pipeline stages. They may be "
        "correct, incomplete, overly generic, duplicated, incorrectly scoped, or "
        "inconsistent with the project description. Do not assume that an existing "
        "goal is correct merely because it is already present.\n\n"

        "For every decision, verify all of the following:\n"
        "1. Whether the original parent is explicitly or semantically supported "
        "by the complete project description.\n"
        "2. Whether the reconstructed candidate is explicitly or semantically "
        "supported by the complete project description.\n"
        "3. Whether the candidate preserves the complete essential functional "
        "intention of its own parent.\n"
        "4. Whether the candidate refers to the same actor or stakeholder.\n"
        "5. Whether the candidate preserves the same functional object or "
        "capability.\n"
        "6. Whether the candidate preserves every essential operation or "
        "responsibility expressed by the parent.\n"
        "7. Whether the candidate preserves the same scope, access level, "
        "visibility, information flow, and intended outcome.\n"
        "8. Whether the candidate introduces unsupported purposes, benefits, "
        "quality attributes, or broader intentions.\n"
        "9. Whether the candidate instead corresponds to another existing "
        "high-level goal.\n"
        "10. Whether the candidate expresses a genuinely new autonomous "
        "functional intention that is documented but not already covered.\n\n"

        "The current branch does not need to represent the entire software system. "
        "It must, however, completely preserve its own documented functional "
        "intention. The global completeness of the entire high-level goal "
        "collection is checked separately by the documentation coverage evaluator.\n\n"

        "Use semantic meaning, not lexical similarity. Consider actor, purpose, "
        "functional object, permitted operations, access level, visibility, "
        "information exchanged, and intended outcome. Two goals are not equivalent "
        "merely because they refer to the same general topic.\n\n"

        "Return exactly one decision:\n"

        "- CONFIRM_BRANCH: choose this decision only if ALL of the following "
        "conditions are satisfied:\n"
        "  1. The candidate refers to the same actor or stakeholder as the parent.\n"
        "  2. The candidate preserves the same functional object or capability.\n"
        "  3. The candidate preserves every essential operation or responsibility "
        "contained in the parent and supported by the project description.\n"
        "  4. The candidate preserves the same functional scope, access level, "
        "visibility, information flow, and intended outcome.\n"
        "  5. The candidate does not introduce unsupported purposes, benefits, "
        "quality attributes, or broader intentions.\n"
        "  6. Both the parent and the candidate are supported by and consistent "
        "with the complete project description.\n"
        "  7. No essential capability present in the parent is absent from the "
        "candidate.\n"
        "If even one of these conditions is not satisfied, do not choose "
        "CONFIRM_BRANCH.\n"

        "- REGENERATE_LOW_LEVEL_GOALS: choose this whenever the candidate is "
        "related to its own parent but fails at least one CONFIRM_BRANCH condition. "
        "This includes cases where the candidate is partial, too generic, "
        "over-broad, differently scoped, or omits one or more essential operations "
        "or responsibilities. Also choose this decision when the candidate replaces "
        "concrete functional capabilities with vague benefits such as efficiency, "
        "engagement, satisfaction, trust, usability, or effectiveness. Preserve "
        "the current parent and request a new low-level decomposition.\n"

        "- MATCHES_OTHER_HIGH_LEVEL_GOAL: choose this when the candidate does not "
        "preserve the intention of its own parent but semantically corresponds to "
        "a DIFFERENT existing high-level goal. The match must also be coherent "
        "with the complete project description, the actor, and the functional "
        "scope. The matched goal must never be the current branch itself.\n"

        "- ADD_NEW_HIGH_LEVEL_GOAL: choose this only when the candidate matches no "
        "existing high-level goal, expresses an autonomous functional intention, "
        "and that intention is explicitly or unambiguously supported by the "
        "complete project description. Before choosing this decision, compare the "
        "candidate semantically with every existing high-level goal. Do not add a "
        "new goal for a minor operation, implementation detail, or subfunction "
        "already covered by a broader existing goal.\n"

        "- REWRITE_ORIGINAL_HIGH_LEVEL_GOAL: choose this only when the complete "
        "project description provides positive evidence that the original parent "
        "is incorrect, unsupported, incomplete, ambiguous, assigned to the wrong "
        "actor, or has an incorrect functional scope. A difference between the "
        "candidate and the parent is not sufficient by itself.\n\n"

        "Decision order:\n"
        "1. Read the complete project description.\n"
        "2. Extract the essential semantic elements of the parent: actor, "
        "functional object, operations, responsibilities, scope, access or "
        "visibility constraints, information flow, and intended outcome.\n"
        "3. Check each essential element explicitly against the candidate.\n"
        "4. If any essential element is missing, altered, unsupported, or replaced "
        "by a generic benefit, choose REGENERATE_LOW_LEVEL_GOALS.\n"
        "5. Choose CONFIRM_BRANCH only if every essential element is preserved.\n"
        "6. If the candidate does not match the parent, compare it with every "
        "other existing high-level goal.\n"
        "7. If it matches another branch, choose "
        "MATCHES_OTHER_HIGH_LEVEL_GOAL.\n"
        "8. If it matches no existing goal but expresses an autonomous documented "
        "intention, consider ADD_NEW_HIGH_LEVEL_GOAL.\n"
        "9. Choose REWRITE_ORIGINAL_HIGH_LEVEL_GOAL only when the project "
        "description positively demonstrates a problem in the original parent.\n\n"

        "Examples:\n\n"

        "Example 1\n"
        "Project description: Customers can create service requests by providing "
        "a category, location, and description.\n"
        "Current parent: Customers can submit service requests and provide the "
        "required information.\n"
        "Candidate: Enable customers to submit service requests with the required "
        "information.\n"
        "Correct decision: CONFIRM_BRANCH.\n"
        "Reason: the candidate preserves the same actor, functional object, "
        "essential operation, and documented scope.\n\n"

        "Example 2\n"
        "Project description: Staff review, approve, assign, and monitor requests.\n"
        "Current parent: Staff review, approve, assign, and monitor requests.\n"
        "Candidate: Staff approve requests.\n"
        "Correct decision: REGENERATE_LOW_LEVEL_GOALS.\n"
        "Reason: the candidate preserves only one documented operation and omits "
        "review, assignment, and monitoring.\n\n"

        "Example 3\n"
        "Project description: Customers submit requests, while technicians manage "
        "assigned interventions.\n"
        "Current branch: Technicians manage assigned interventions.\n"
        "Candidate: Customers submit new service requests.\n"
        "Existing goals:\n"
        "- branch_001: Customers submit new service requests.\n"
        "- branch_003: Technicians manage assigned interventions.\n"
        "Correct decision: MATCHES_OTHER_HIGH_LEVEL_GOAL.\n"
        "matched_high_level_goal_id: branch_001.\n"
        "Reason: the candidate is documented but belongs to another existing "
        "branch and actor.\n\n"

        "Example 4\n"
        "Project description: Citizens receive notifications for every report "
        "status change and can exchange messages with municipal operators.\n"
        "Current parent: Citizens receive status notifications and communicate "
        "with municipal operators.\n"
        "Candidate: Improve citizen engagement through personalized report "
        "updates.\n"
        "Correct decision: REGENERATE_LOW_LEVEL_GOALS.\n"
        "Reason: the candidate is thematically related but omits bidirectional "
        "communication with municipal operators and replaces concrete capabilities "
        "with a generic benefit.\n\n"

        "Hard rules:\n"
        "- The complete project description is the primary source of truth for "
        "every decision.\n"
        "- Consider the entire project description, not only the most similar "
        "sentence or section.\n"
        "- CONFIRM_BRANCH is an all-or-nothing decision: every mandatory condition "
        "must hold.\n"
        "- When uncertain between CONFIRM_BRANCH and "
        "REGENERATE_LOW_LEVEL_GOALS, choose REGENERATE_LOW_LEVEL_GOALS.\n"
        "- A candidate that preserves only the general topic but omits one "
        "concrete functional responsibility must not be confirmed.\n"
        "- A candidate that replaces operations with abstract benefits such as "
        "efficiency, engagement, trust, satisfaction, usability, or effectiveness "
        "must not be confirmed unless those benefits are themselves the actual "
        "documented functional intention.\n"
        "- A shorter or more abstract candidate may be confirmed only when its "
        "wording still semantically entails every essential capability of the "
        "parent.\n"
        "- Do not infer missing capabilities from the abstraction rationale. "
        "Evaluate the reconstructed candidate goal itself.\n"
        "- Do not use cohesion or confidence to compensate for missing semantic "
        "content. High cohesion or confidence does not justify CONFIRM_BRANCH.\n"
        "- matched_high_level_goal_id must never equal the current branch_id.\n"
        "- Do not add a new goal before comparing the candidate with every "
        "existing high-level goal.\n"
        "- Do not treat a minor subfunction as an autonomous high-level goal when "
        "it is already covered by an existing broader intention.\n"
        "- Do not invent functionality, actors, constraints, quality attributes, "
        "or domain facts.\n"
        "- Do not generate or evaluate individual low-level goals.\n"
        "- Generate only functional goals.\n"
        "- For ADD_NEW_HIGH_LEVEL_GOAL and "
        "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL, provide both a non-empty name and "
        "description.\n"
        "- For all other decisions, those fields must be null.\n"
        "- matched_high_level_goal_id must be null except for "
        "MATCHES_OTHER_HIGH_LEVEL_GOAL.\n"
        "- The rationale must explicitly state which essential elements were "
        "preserved, omitted, changed, or unsupported.\n"
        "- Return only valid JSON, without Markdown or additional text.\n\n"

        "Output structure:\n"
        "{\n"
        '  "decision": "CONFIRM_BRANCH | REGENERATE_LOW_LEVEL_GOALS | '
        'MATCHES_OTHER_HIGH_LEVEL_GOAL | ADD_NEW_HIGH_LEVEL_GOAL | '
        'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",\n'
        '  "matched_high_level_goal_id": "string or null",\n'
        '  "new_or_replacement_goal_name": "string or null",\n'
        '  "new_or_replacement_goal_description": "string or null",\n'
        '  "rationale": "string"\n'
        "}\n\n"

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

    # matched_high_level_goal_id ha senso solo per la decisione di "match"
    # con un altro obiettivo già esistente: per ogni altra decisione deve
    # restare null.
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

    # Nome/descrizione del nuovo obiettivo hanno senso solo quando si
    # aggiunge un obiettivo nuovo o si riscrive quello originale.
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

    # Rimuove eventuali fence Markdown (```...```) che il modello può aver
    # aggiunto attorno al JSON, così il parsing sotto non fallisce.
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
    empty_branches: list[str],
) -> tuple[dict[str, GlobalGoalEvaluationResult], dict[str, str]]:
    """
    Produces one evaluation for every expected branch.

    Branches with a bottom-up candidate are evaluated through the LLM.
    Branches with no low-level goals receive a deterministic
    REGENERATE_LOW_LEVEL_GOALS decision without an LLM call.

    A genuine reconstruction failure is not converted into an empty-branch
    decision: the corresponding evaluation remains missing and the final JSON
    is marked INCOMPLETE_EVALUATION.
    """
    results: dict[str, GlobalGoalEvaluationResult] = {}
    errors: dict[str, str] = {}

    expected_branch_ids = set(existing_high_level_goals)
    reconstructed_branch_ids = set(reconstructed_high_level_goals)
    empty_branch_ids = set(empty_branches)

    # Controlli di coerenza tra gli insiemi di branch attesi, ricostruiti
    # e vuoti, prima di valutare qualunque branch.

    # Un branch ricostruito che non fa parte di quelli attesi è un errore.
    unexpected_reconstructed = (
        reconstructed_branch_ids - expected_branch_ids
    )
    for branch_id in sorted(unexpected_reconstructed):
        errors[branch_id] = (
            "GlobalGoalEvaluationError: a bottom-up reconstruction was "
            "provided for an unexpected branch."
        )

    # Un branch marcato come vuoto ma sconosciuto è un errore.
    unexpected_empty = empty_branch_ids - expected_branch_ids
    for branch_id in sorted(unexpected_empty):
        errors[branch_id] = (
            "GlobalGoalEvaluationError: an unknown branch was marked empty."
        )

    # Un branch non può essere sia ricostruito sia vuoto allo stesso tempo.
    overlap = reconstructed_branch_ids & empty_branch_ids
    for branch_id in sorted(overlap):
        errors[branch_id] = (
            "GlobalGoalEvaluationError: the branch cannot simultaneously "
            "have a reconstructed candidate and be marked as empty."
        )

    for branch_id, original in existing_high_level_goals.items():
        if branch_id in overlap:
            continue

        reconstructed = reconstructed_high_level_goals.get(branch_id)

        if reconstructed is not None:
            try:
                results[branch_id] = evaluate_branch(
                    branch_id=branch_id,
                    project_description=project_description,
                    original_high_level_goal=original,
                    reconstructed_high_level_goal=reconstructed,
                    existing_high_level_goals=existing_high_level_goals,
                )
            except Exception as exc:
                errors[branch_id] = f"{type(exc).__name__}: {exc}"

            continue

        if branch_id in empty_branch_ids:
            results[branch_id] = build_empty_branch_evaluation(
                branch_id=branch_id,
                original_high_level_goal=original,
            )
            continue

        errors[branch_id] = (
            "GlobalGoalEvaluationError: the expected branch has neither "
            "a reconstructed high-level goal nor an empty-branch marker. "
            "This indicates a missing or failed bottom-up reconstruction."
        )

    return results, errors

def build_empty_branch_evaluation(
    branch_id: str,
    original_high_level_goal: HighLevelGoal,
) -> GlobalGoalEvaluationResult:
    """
    Builds a deterministic evaluation for a branch with no low-level goals.

    No LLM call is performed because no bottom-up candidate exists to compare.
    The original parent is preserved and its low-level decomposition must be
    generated.
    """
    return GlobalGoalEvaluationResult(
        branch_id=branch_id,
        decision="REGENERATE_LOW_LEVEL_GOALS",
        original_high_level_goal=original_high_level_goal,
        reconstructed_high_level_goal=None,
        matched_high_level_goal=None,
        new_or_replacement_high_level_goal=None,
        rationale=(
            "The branch contains no low-level goals, so no bottom-up "
            "high-level goal could be reconstructed. The original parent "
            "is preserved and its low-level decomposition must be generated."
        ),
        operational_action=(
            _OPERATIONAL_ACTION_BY_DECISION[
                "REGENERATE_LOW_LEVEL_GOALS"
            ]
        ),
        requires_low_level_regeneration=True,
    )

def _utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp for persisted evaluator outputs."""
    return datetime.now(timezone.utc).isoformat()


def save_global_evaluations(
    output_file: str | Path,
    expected_high_level_goals: dict[str, HighLevelGoal],
    evaluations: dict[str, GlobalGoalEvaluationResult],
    errors: dict[str, str],
) -> Path:
    """
    Persist one complete branch-level evaluation JSON for orchestration.

    The JSON is marked READY_FOR_ORCHESTRATION only when:
    - every expected branch has exactly one evaluation;
    - no unexpected branch is present;
    - every dictionary key matches evaluation.branch_id;
    - no evaluator error is present.

    Incomplete or inconsistent outputs are still persisted for diagnostics,
    but the orchestrator must reject them and apply no decision.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    expected_branch_ids = list(expected_high_level_goals.keys())
    expected_set = set(expected_branch_ids)
    evaluated_set = set(evaluations.keys())

    # Confronta l'insieme dei branch attesi con quelli effettivamente
    # valutati, per capire cosa manca o cosa è di troppo.
    missing_branch_ids = sorted(expected_set - evaluated_set)
    unexpected_branch_ids = sorted(evaluated_set - expected_set)

    inconsistent_branch_ids = sorted(
        branch_id
        for branch_id, evaluation in evaluations.items()
        if evaluation.branch_id != branch_id
    )

    completeness_errors = dict(errors)

    for branch_id in missing_branch_ids:
        completeness_errors.setdefault(
            branch_id,
            "GlobalGoalEvaluationError: no complete evaluation was produced "
            "for this expected branch.",
        )

    for branch_id in unexpected_branch_ids:
        completeness_errors.setdefault(
            branch_id,
            "GlobalGoalEvaluationError: an evaluation was produced for an "
            "unexpected branch.",
        )

    for branch_id in inconsistent_branch_ids:
        completeness_errors.setdefault(
            branch_id,
            "GlobalGoalEvaluationError: the evaluation dictionary key does "
            "not match evaluation.branch_id.",
        )

    is_complete = (
        not completeness_errors
        and not missing_branch_ids
        and not unexpected_branch_ids
        and not inconsistent_branch_ids
        and expected_set == evaluated_set
    )

    payload = {
        "schema_version": "1.0",
        "status": (
            "READY_FOR_ORCHESTRATION"
            if is_complete
            else "INCOMPLETE_EVALUATION"
        ),
        "expected_branch_ids": expected_branch_ids,
        "evaluated_branch_ids": sorted(evaluated_set),
        "missing_branch_ids": missing_branch_ids,
        "unexpected_branch_ids": unexpected_branch_ids,
        "inconsistent_branch_ids": inconsistent_branch_ids,
        "evaluations": {
            branch_id: evaluation.model_dump(mode="json")
            for branch_id, evaluation in evaluations.items()
        },
        "errors": completeness_errors,
        "created_at_utc": _utc_timestamp(),
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Global documentation coverage evaluator
# ---------------------------------------------------------------------------


class DocumentationCoverageEvaluationError(ValueError):
    """Raised when the documentation coverage LLM output is invalid."""


def _build_documentation_coverage_system_prompt() -> str:
    return (
        "You are a helpful assistant, expert in Software Engineering, "
        "Requirements Engineering, and specialised in Goal-Oriented "
        "Requirements Engineering (GORE).\n\n"

        "You are performing a GLOBAL DOCUMENTATION COVERAGE ANALYSIS. "
        "You are given the complete project description and the complete "
        "current collection of high-level functional goals.\n\n"

        "The complete project description is the primary source of truth. "
        "Read and consider the entire description before producing the result. "
        "Do not focus only on isolated keywords, individual sentences, or the "
        "sections most similar to the existing goals.\n\n"

        "The current high-level goals were generated by previous pipeline stages "
        "and may be incomplete, duplicated, overly generic, incorrectly scoped, "
        "or assigned to an incorrect actor. Do not assume that the current "
        "collection is correct merely because the goals already exist.\n\n"

        "The complete current high-level goal collection must be COMPLETELY "
        "representative of all autonomous functional intentions explicitly or "
        "unambiguously supported by the complete project description.\n\n"

        "Determine whether every autonomous functional intention supported by the "
        "project description is represented by at least one existing high-level "
        "goal. Use semantic coverage, not lexical overlap.\n\n"

        "Evaluate coverage by considering:\n"
        "- the responsible actor or stakeholder;\n"
        "- the functional purpose;\n"
        "- the operations or capabilities allowed;\n"
        "- access level and visibility;\n"
        "- the information exchanged or managed;\n"
        "- the intended outcome.\n\n"

        "A documented capability does not require a separate high-level goal when "
        "it is only a low-level operation, implementation detail, or subfunction "
        "already semantically covered by a broader existing high-level goal.\n\n"

        "Return exactly one status:\n"
        "- COMPLETE: the complete existing HLG collection represents every "
        "autonomous functional intention supported by the documentation.\n"
        "- MISSING_HIGH_LEVEL_GOALS: one or more autonomous documented functional "
        "intentions are genuinely absent from the existing collection.\n\n"

        "For every missing high-level goal provide its name, description, actor, "
        "concise source_evidence grounded in the project description, and a "
        "rationale explaining why no existing goal already covers it.\n\n"

        "Hard rules:\n"
        "- Treat the complete project description as the primary source of truth.\n"
        "- Consider the entire project description before determining coverage.\n"
        "- Before proposing a missing goal, compare its intention semantically "
        "with every existing high-level goal.\n"
        "- Propose a new goal only when an autonomous documented functional "
        "intention remains genuinely uncovered.\n"
        "- Do not propose a new goal merely because the documentation describes "
        "additional details, channels, fields, steps, or variants of an intention "
        "already covered by an existing goal.\n"
        "- Distinguish intentions that differ materially in actor, access level, "
        "visibility, purpose, permitted operations, or intended outcome.\n"
        "- Do not combine different stakeholder categories into a single actor "
        "unless the documentation explicitly assigns them the same intention and "
        "the same functional scope.\n"
        "- Never invent functionality, actors, constraints, quality attributes, "
        "or system behaviour.\n"
        "- Generate only functional high-level goals, not low-level operations.\n"
        "- Do not rewrite or delete existing high-level goals.\n"
        "- Do not propose a goal that is semantically equivalent to or already "
        "subsumed by an existing goal.\n"
        "- Do not duplicate proposed missing goals.\n"
        "- If status is COMPLETE, missing_high_level_goals must be empty.\n"
        "- If status is MISSING_HIGH_LEVEL_GOALS, missing_high_level_goals must "
        "contain at least one item.\n"
        "- Respond only with valid JSON, without Markdown or code fences.\n\n"

        "Return exactly this structure:\n"
        "{\n"
        '  "status": "COMPLETE | MISSING_HIGH_LEVEL_GOALS",\n'
        '  "missing_high_level_goals": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "description": "string",\n'
        '      "actor": {"name": "string", "description": "string"},\n'
        '      "source_evidence": "string",\n'
        '      "rationale": "string"\n'
        "    }\n"
        "  ],\n"
        '  "observations": ["string"]\n'
        "}\n\n"

        "Respond only with the JSON object."
    )


def _build_documentation_coverage_user_prompt(
    project_description: str,
    current_high_level_goals: HighLevelGoals,
) -> str:
    goals_block = "\n".join(
        f"- [{index}] {goal.name}: {goal.description} "
        f"(actor: {goal.actor.name})"
        for index, goal in enumerate(current_high_level_goals.goals, start=1)
    )

    return (
        "**Project description:**\n"
        f"{project_description}\n\n"
        "**Complete current high-level goal collection:**\n"
        f"{goals_block}\n\n"
        "Check whether the current collection covers every autonomous functional "
        "intention explicitly supported by the project description.\n\n"
        "**Output:**"
    )


def _parse_documentation_coverage(
    raw_response: str,
) -> DocumentationCoverageLLMOutput:
    if not raw_response or not raw_response.strip():
        raise DocumentationCoverageEvaluationError(
            "The documentation coverage evaluator returned an empty response."
        )

    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise DocumentationCoverageEvaluationError(
            "The documentation coverage evaluator did not return valid JSON. "
            f"Response received: {raw_response}"
        ) from exc

    try:
        return DocumentationCoverageLLMOutput.model_validate(payload)
    except Exception as exc:
        raise DocumentationCoverageEvaluationError(
            "The documentation coverage response does not respect the expected "
            "DocumentationCoverageLLMOutput structure. "
            f"Response received: {raw_response}"
        ) from exc


def _validate_documentation_coverage_output(
    output: DocumentationCoverageLLMOutput,
) -> None:
    if output.status == "COMPLETE" and output.missing_high_level_goals:
        raise DocumentationCoverageEvaluationError(
            "Status COMPLETE cannot contain missing high-level goals."
        )

    if (
        output.status == "MISSING_HIGH_LEVEL_GOALS"
        and not output.missing_high_level_goals
    ):
        raise DocumentationCoverageEvaluationError(
            "Status MISSING_HIGH_LEVEL_GOALS requires at least one missing goal."
        )


def evaluate_documentation_coverage(
    project_description: str,
    current_high_level_goals: HighLevelGoals,
) -> DocumentationCoverageResult:
    """Compares the entire HLG collection with the complete documentation."""
    raw_response = generate_response_llama(
        _build_documentation_coverage_user_prompt(
            project_description,
            current_high_level_goals,
        ),
        _build_documentation_coverage_system_prompt(),
    )

    output = _parse_documentation_coverage(raw_response)
    _validate_documentation_coverage_output(output)

    seen_normalized_names = {
        normalize_goal_name(goal.name)
        for goal in current_high_level_goals.goals
    }
    added_high_level_goals: list[HighLevelGoal] = []

    for proposal in output.missing_high_level_goals:
        normalized_name = normalize_goal_name(proposal.name)
        if normalized_name in seen_normalized_names:
            continue

        seen_normalized_names.add(normalized_name)
        added_high_level_goals.append(
            HighLevelGoal(
                name=proposal.name,
                description=proposal.description,
                actor=proposal.actor,
            )
        )

    if (
        output.status == "MISSING_HIGH_LEVEL_GOALS"
        and not added_high_level_goals
    ):
        raise DocumentationCoverageEvaluationError(
            "Missing goals were reported, but every proposal duplicated an "
            "existing or previously proposed high-level goal."
        )

    return DocumentationCoverageResult(
        status=output.status,
        missing_high_level_goals=output.missing_high_level_goals,
        added_high_level_goals=added_high_level_goals,
        observations=output.observations,
    )



def save_documentation_coverage(
    output_file: str | Path,
    coverage_result: DocumentationCoverageResult,
) -> Path:
    """Persist the global documentation-coverage evaluation without applying it."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "status": "READY_FOR_ORCHESTRATION",
        "documentation_coverage": coverage_result.model_dump(mode="json"),
        "created_at_utc": _utc_timestamp(),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
