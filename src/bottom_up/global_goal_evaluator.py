"""
Global Goal Evaluator.

For each branch, this module compares the bottom-up reconstructed high-level
candidate with its original parent, the complete set of current high-level
goals, and the project description.

The evaluator is deliberately separated from the existing top-down HLG
generator:
- it decides whether a branch is correct or requires revision;
- when an HLG must be added or replaced, it prepares a normal generation
  request for the existing top-down generator;
- it never generates the final HighLevelGoal object;
- it never applies changes to the HLG collection;
- it never regenerates low-level goals.

Only ``generation_request.generator_input`` is forwarded to the top-down HLG
generator. That input contains the same two arguments used during the initial
top-down execution: a project description and a set of actors. The generator
is therefore not informed about evaluations, corrections, replacements,
branch identifiers, or previous attempts.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.data_model import (
    Actors,
    BottomUpHighLevelGoal,
    DocumentationCoverageLLMOutput,
    DocumentationCoverageResult,
    GlobalGoalEvaluationLLMOutput,
    GlobalGoalEvaluationResult,
    HighLevelGoal,
    HighLevelGoalGenerationRequest,
    HighLevelGoalGeneratorInput,
    HighLevelGoals,
)
from src.bottom_up.goal_reconstructor import normalize_goal_name
from src.llm_clients import generate_response_llama


class GlobalGoalEvaluationError(ValueError):
    """Raised when a branch-evaluation response is invalid or inconsistent."""


_OPERATIONAL_ACTION_BY_DECISION: dict[str, str] = {
    "CONFIRM_BRANCH": "KEEP_PARENT",
    "REGENERATE_LOW_LEVEL_GOALS": "KEEP_PARENT_AND_REGENERATE_LOW_LEVEL_GOALS",
    "MATCHES_OTHER_HIGH_LEVEL_GOAL": (
        "KEEP_PARENT_REGENERATE_LOW_LEVEL_GOALS_AND_RECORD_MISASSIGNMENT"
    ),
    "ADD_NEW_HIGH_LEVEL_GOAL": (
        "GENERATE_NEW_HIGH_LEVEL_GOAL_AND_REGENERATE_LOW_LEVEL_GOALS"
    ),
    "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL": (
        "GENERATE_REPLACEMENT_HIGH_LEVEL_GOAL_AND_REGENERATE_LOW_LEVEL_GOALS"
    ),
}

_HLG_GENERATION_DECISIONS = {
    "ADD_NEW_HIGH_LEVEL_GOAL",
    "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",
}


def _build_system_prompt() -> str:
    return (
        "You are an expert in Software Engineering, Requirements Engineering, "
        "and Goal-Oriented Requirements Engineering (GORE).\n\n"

        "You are the GLOBAL GOAL EVALUATOR. Evaluate one high-level-goal "
        "branch and return exactly one structured decision.\n\n"

        "A separate top-down component generates high-level goals. You must "
        "never generate a final high-level-goal name or description. When your "
        "decision requires a new or replacement HLG, provide only a focused, "
        "self-contained project description in generation_project_description. "
        "That description will be passed to the normal top-down HLG generator "
        "together with the relevant actor, exactly as in an initial generation.\n\n"

        "The generation_project_description field must read like ordinary "
        "stakeholder documentation. It must describe the required functional "
        "intention directly and naturally. It must not mention evaluation, "
        "feedback, correction, rewriting, replacement, addition, missing goals, "
        "existing goals, branches, previous attempts, or the current pipeline.\n\n"

        "You must compare:\n"
        "- the original parent high-level goal;\n"
        "- the candidate high-level goal reconstructed bottom-up;\n"
        "- the complete set of current high-level goals;\n"
        "- the complete project description.\n\n"

        "The complete project description is the primary source of truth. Read "
        "and consider the entire description before deciding. Existing goals and "
        "the reconstructed candidate may be correct, incomplete, overly generic, "
        "duplicated, incorrectly scoped, or unsupported.\n\n"

        "For every decision, verify all of the following:\n"
        "1. Whether the original parent is supported by the documentation.\n"
        "2. Whether the reconstructed candidate is supported by the documentation.\n"
        "3. Whether the candidate preserves the complete functional intention of "
        "its own parent.\n"
        "4. Whether the candidate preserves the actor, functional object, "
        "operations, responsibilities, scope, access level, visibility, "
        "information flow, and intended outcome.\n"
        "5. Whether the candidate introduces unsupported purposes, benefits, "
        "quality attributes, actors, or broader intentions.\n"
        "6. Whether the candidate instead corresponds to another existing HLG.\n"
        "7. Whether the candidate expresses a genuinely new autonomous functional "
        "intention documented but absent from the current HLG collection.\n\n"

        "Return exactly one decision:\n\n"

        "- CONFIRM_BRANCH: choose this only when the candidate preserves every "
        "essential semantic element of its original parent and both are supported "
        "by the complete project description. generation_project_description "
        "must be null.\n\n"

        "- REGENERATE_LOW_LEVEL_GOALS: choose this when the original parent "
        "remains correct, but the bottom-up candidate reveals that the current "
        "low-level decomposition is partial, too generic, over-broad, differently "
        "scoped, or missing essential responsibilities. Preserve the parent and "
        "request only a new low-level decomposition. "
        "generation_project_description must be null.\n\n"

        "- MATCHES_OTHER_HIGH_LEVEL_GOAL: choose this when the candidate does not "
        "preserve its own parent but semantically corresponds to a different "
        "existing HLG. matched_high_level_goal_id must identify that different "
        "branch. generation_project_description must be null.\n\n"

        "- ADD_NEW_HIGH_LEVEL_GOAL: choose this only when the candidate matches no "
        "existing HLG, expresses an autonomous functional intention associated "
        "with the current branch actor, and that intention is explicitly or "
        "unambiguously supported by the documentation. The current parent is not "
        "replaced. generation_project_description must contain a focused, "
        "self-contained stakeholder description of only that additional "
        "functional intention. Do not state that a goal must be added and do not "
        "refer to the existing HLG collection.\n\n"

        "- REWRITE_ORIGINAL_HIGH_LEVEL_GOAL: choose this only when the complete "
        "documentation positively demonstrates that the original parent is "
        "incorrect, incomplete, ambiguous, unsupported, or incorrectly scoped, "
        "while the responsible actor remains the current branch actor. "
        "generation_project_description must contain a focused, self-contained "
        "stakeholder description of the correct documented functional intention. "
        "Do not mention the old goal, its defects, rewriting, replacement, or "
        "evaluation.\n\n"

        "Rules for generation_project_description:\n"
        "- Use it only for ADD_NEW_HIGH_LEVEL_GOAL or "
        "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL.\n"
        "- Describe exactly one autonomous functional intention.\n"
        "- Include only information supported by the complete project description.\n"
        "- Preserve the relevant actor, functional object, operations, scope, "
        "access level, visibility, information flow, and intended outcome.\n"
        "- Write ordinary project documentation, not an instruction or critique.\n"
        "- Do not include a final HLG name.\n"
        "- Do not mention any branch identifier.\n"
        "- Do not mention current or previous goals.\n"
        "- Do not mention addition, correction, rewriting, or replacement.\n"
        "- Do not introduce unsupported functionality, implementation details, "
        "quality attributes, or domain facts.\n\n"

        "Decision order:\n"
        "1. Read the complete project description.\n"
        "2. Extract the essential semantic elements of the parent.\n"
        "3. Compare every essential element with the candidate.\n"
        "4. If the parent is correct but the candidate is partial or distorted, "
        "choose REGENERATE_LOW_LEVEL_GOALS.\n"
        "5. Choose CONFIRM_BRANCH only when every essential element is preserved.\n"
        "6. If the candidate does not match the parent, compare it with every "
        "other existing HLG.\n"
        "7. If it matches another branch, choose MATCHES_OTHER_HIGH_LEVEL_GOAL.\n"
        "8. If it matches no branch but represents an autonomous documented "
        "intention for the same actor, consider ADD_NEW_HIGH_LEVEL_GOAL.\n"
        "9. Choose REWRITE_ORIGINAL_HIGH_LEVEL_GOAL only when the documentation "
        "provides positive evidence that the original parent itself is defective.\n\n"

        "Hard rules:\n"
        "- Use semantic meaning, not lexical similarity.\n"
        "- CONFIRM_BRANCH is an all-or-nothing decision.\n"
        "- When uncertain between CONFIRM_BRANCH and "
        "REGENERATE_LOW_LEVEL_GOALS, choose REGENERATE_LOW_LEVEL_GOALS.\n"
        "- Do not infer missing capabilities from the abstraction rationale; "
        "evaluate the reconstructed candidate itself.\n"
        "- Do not use cohesion or confidence to compensate for missing semantic "
        "content.\n"
        "- matched_high_level_goal_id must never equal the current branch_id.\n"
        "- Compare the candidate with every existing HLG before adding a new one.\n"
        "- Do not treat a minor operation or implementation detail as an "
        "autonomous HLG.\n"
        "- Do not generate or evaluate individual low-level goals.\n"
        "- The rationale must state which essential elements were preserved, "
        "omitted, changed, unsupported, or assigned elsewhere.\n"
        "- Return only valid JSON, without Markdown or additional text.\n\n"

        "Output structure:\n"
        "{\n"
        '  "decision": "CONFIRM_BRANCH | REGENERATE_LOW_LEVEL_GOALS | '
        'MATCHES_OTHER_HIGH_LEVEL_GOAL | ADD_NEW_HIGH_LEVEL_GOAL | '
        'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",\n'
        '  "matched_high_level_goal_id": "string or null",\n'
        '  "generation_project_description": "string or null",\n'
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
        f"**Current branch_id:** {branch_id}\n\n"
        "**Complete project description:**\n"
        f"{project_description}\n\n"
        "**Original parent high-level goal:**\n"
        f"- Actor: {original_high_level_goal.actor.name} - "
        f"{original_high_level_goal.actor.description}\n"
        f"- Name: {original_high_level_goal.name}\n"
        f"- Description: {original_high_level_goal.description}\n\n"
        "**Bottom-up reconstructed candidate for this branch:**\n"
        f"- Reconstructed goal: "
        f"{reconstructed_high_level_goal.reconstructed_high_level_goal}\n"
        f"- Abstraction rationale: "
        f"{reconstructed_high_level_goal.abstraction_rationale}\n"
        f"- Cohesion: {reconstructed_high_level_goal.cohesion}\n"
        f"- Confidence: {reconstructed_high_level_goal.confidence}\n\n"
        "**Complete current set of high-level goals:**\n"
        f"{existing_block}\n\n"
        "Evaluate this branch. When HLG generation is required, return a "
        "focused stakeholder-style project description for the normal top-down "
        "generator, not a final goal and not a correction instruction.\n\n"
        "**Output:**"
    )


def _resolve_matched_goal(
    branch_id: str,
    matched_high_level_goal_id: str | None,
    existing_high_level_goals: dict[str, HighLevelGoal],
) -> HighLevelGoal:
    if not matched_high_level_goal_id:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': MATCHES_OTHER_HIGH_LEVEL_GOAL requires "
            "matched_high_level_goal_id."
        )

    if matched_high_level_goal_id == branch_id:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': MATCHES_OTHER_HIGH_LEVEL_GOAL cannot "
            "reference the current branch itself."
        )

    matched = existing_high_level_goals.get(matched_high_level_goal_id)
    if matched is None:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': unknown matched_high_level_goal_id "
            f"'{matched_high_level_goal_id}'. Valid ids are: "
            f"{sorted(existing_high_level_goals.keys())}."
        )

    return matched


def _build_generation_request(
    branch_id: str,
    llm_output: GlobalGoalEvaluationLLMOutput,
    original_high_level_goal: HighLevelGoal,
) -> HighLevelGoalGenerationRequest:
    """
    Convert an evaluator decision into a request compatible with the original
    top-down HLG generator.

    Only ``generator_input`` is forwarded to the generator. Action and branch
    identifiers remain private orchestration metadata.
    """
    if llm_output.decision not in _HLG_GENERATION_DECISIONS:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': decision '{llm_output.decision}' does not "
            "require an HLG generation request."
        )

    focused_description = llm_output.generation_project_description
    if not focused_description or not focused_description.strip():
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': decision '{llm_output.decision}' requires "
            "a non-empty generation_project_description."
        )

    if llm_output.decision == "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL":
        action = "REPLACE_EXISTING_HIGH_LEVEL_GOAL"
        request_id = f"{branch_id}_replace_existing_hlg"
        target_branch_id = branch_id
    else:
        action = "ADD_NEW_HIGH_LEVEL_GOAL"
        request_id = f"{branch_id}_add_new_hlg"
        target_branch_id = None

    return HighLevelGoalGenerationRequest(
        request_id=request_id,
        action=action,
        source="BRANCH_EVALUATION",
        generator_input=HighLevelGoalGeneratorInput(
            project_description=focused_description,
            actors=Actors(actors=[original_high_level_goal.actor]),
        ),
        target_branch_id=target_branch_id,
        origin_branch_id=branch_id,
        rationale=llm_output.rationale,
    )


def _parse_llama_evaluation(
    raw_response: str,
) -> GlobalGoalEvaluationLLMOutput:
    """Parse and validate the JSON returned by the evaluator LLM."""
    if not raw_response or not raw_response.strip():
        raise GlobalGoalEvaluationError(
            "Llama returned an empty Global Goal Evaluation response."
        )

    cleaned_response = raw_response.strip()
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
            "Llama returned JSON that does not respect "
            "GlobalGoalEvaluationLLMOutput. "
            f"Response received: {raw_response}"
        ) from exc


def evaluate_branch(
    branch_id: str,
    project_description: str,
    original_high_level_goal: HighLevelGoal,
    reconstructed_high_level_goal: BottomUpHighLevelGoal,
    existing_high_level_goals: dict[str, HighLevelGoal],
) -> GlobalGoalEvaluationResult:
    """Evaluate one branch without generating or applying a final HLG."""
    raw_llm_output = generate_response_llama(
        _build_user_prompt(
            branch_id=branch_id,
            project_description=project_description,
            original_high_level_goal=original_high_level_goal,
            reconstructed_high_level_goal=reconstructed_high_level_goal,
            existing_high_level_goals=existing_high_level_goals,
        ),
        _build_system_prompt(),
    )

    llm_output = _parse_llama_evaluation(raw_llm_output)
    decision = llm_output.decision

    matched_high_level_goal: HighLevelGoal | None = None
    generation_request: HighLevelGoalGenerationRequest | None = None

    if decision == "MATCHES_OTHER_HIGH_LEVEL_GOAL":
        matched_high_level_goal = _resolve_matched_goal(
            branch_id=branch_id,
            matched_high_level_goal_id=llm_output.matched_high_level_goal_id,
            existing_high_level_goals=existing_high_level_goals,
        )
    elif decision in _HLG_GENERATION_DECISIONS:
        generation_request = _build_generation_request(
            branch_id=branch_id,
            llm_output=llm_output,
            original_high_level_goal=original_high_level_goal,
        )
    elif decision not in {
        "CONFIRM_BRANCH",
        "REGENERATE_LOW_LEVEL_GOALS",
    }:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': unsupported decision '{decision}'."
        )

    requires_high_level_regeneration = decision in _HLG_GENERATION_DECISIONS

    return GlobalGoalEvaluationResult(
        branch_id=branch_id,
        decision=decision,
        original_high_level_goal=original_high_level_goal,
        reconstructed_high_level_goal=(
            reconstructed_high_level_goal.reconstructed_high_level_goal
        ),
        matched_high_level_goal=matched_high_level_goal,
        generation_request=generation_request,
        rationale=llm_output.rationale,
        operational_action=_OPERATIONAL_ACTION_BY_DECISION[decision],
        requires_high_level_regeneration=requires_high_level_regeneration,
        requires_low_level_regeneration=decision != "CONFIRM_BRANCH",
    )


def evaluate_all_branches(
    project_description: str,
    existing_high_level_goals: dict[str, HighLevelGoal],
    reconstructed_high_level_goals: dict[str, BottomUpHighLevelGoal],
    empty_branches: list[str],
) -> tuple[dict[str, GlobalGoalEvaluationResult], dict[str, str]]:
    """Produce exactly one validated evaluation for every expected branch."""
    results: dict[str, GlobalGoalEvaluationResult] = {}
    errors: dict[str, str] = {}

    expected_branch_ids = set(existing_high_level_goals)
    reconstructed_branch_ids = set(reconstructed_high_level_goals)
    empty_branch_ids = set(empty_branches)

    unexpected_reconstructed = reconstructed_branch_ids - expected_branch_ids
    for branch_id in sorted(unexpected_reconstructed):
        errors[branch_id] = (
            "GlobalGoalEvaluationError: a bottom-up reconstruction was "
            "provided for an unexpected branch."
        )

    unexpected_empty = empty_branch_ids - expected_branch_ids
    for branch_id in sorted(unexpected_empty):
        errors[branch_id] = (
            "GlobalGoalEvaluationError: an unknown branch was marked empty."
        )

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
            "a reconstructed HLG nor an empty-branch marker. This indicates "
            "a missing or failed bottom-up reconstruction."
        )

    return results, errors


def build_empty_branch_evaluation(
    branch_id: str,
    original_high_level_goal: HighLevelGoal,
) -> GlobalGoalEvaluationResult:
    """Build the deterministic evaluation for an HLG with no current LLGs."""
    return GlobalGoalEvaluationResult(
        branch_id=branch_id,
        decision="REGENERATE_LOW_LEVEL_GOALS",
        original_high_level_goal=original_high_level_goal,
        reconstructed_high_level_goal=None,
        matched_high_level_goal=None,
        generation_request=None,
        rationale=(
            "The branch contains no low-level goals, so no bottom-up HLG could "
            "be reconstructed. The original parent is preserved and its "
            "low-level decomposition must be generated."
        ),
        operational_action=(
            _OPERATIONAL_ACTION_BY_DECISION["REGENERATE_LOW_LEVEL_GOALS"]
        ),
        requires_high_level_regeneration=False,
        requires_low_level_regeneration=True,
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_global_evaluations(
    output_file: str | Path,
    expected_high_level_goals: dict[str, HighLevelGoal],
    evaluations: dict[str, GlobalGoalEvaluationResult],
    errors: dict[str, str],
) -> Path:
    """Persist one complete branch-level evaluator JSON for orchestration."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    expected_branch_ids = list(expected_high_level_goals.keys())
    expected_set = set(expected_branch_ids)
    evaluated_set = set(evaluations.keys())

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
        "schema_version": "3.0",
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
    """Raised when the documentation-coverage response is invalid."""


def _build_documentation_coverage_system_prompt() -> str:
    return (
        "You are an expert in Software Engineering, Requirements Engineering, "
        "and Goal-Oriented Requirements Engineering (GORE).\n\n"

        "You are performing a GLOBAL DOCUMENTATION COVERAGE ANALYSIS. You are "
        "given the complete project description and the complete current HLG "
        "collection.\n\n"

        "The complete project description is the primary source of truth. "
        "Determine whether every autonomous functional intention supported by "
        "the documentation is represented by at least one current HLG. Use "
        "semantic coverage, not lexical overlap.\n\n"

        "You are an evaluator, not the final HLG generator. When an autonomous "
        "functional intention is missing, provide a focused, self-contained "
        "project_description that can be passed as ordinary input to the normal "
        "top-down HLG generator together with the responsible actor. Do not "
        "return a final goal name or a final goal description.\n\n"

        "The project_description field must read like normal stakeholder "
        "documentation. It must describe only the missing functional intention. "
        "It must not mention coverage analysis, evaluation, missing goals, "
        "addition, correction, existing goals, previous attempts, or the "
        "pipeline.\n\n"

        "Evaluate coverage by considering actor, functional purpose, operations, "
        "capabilities, access level, visibility, information flow, and intended "
        "outcome. A minor operation, implementation detail, field, channel, step, "
        "or variant already subsumed by an existing HLG is not a missing HLG.\n\n"

        "Return exactly one status:\n"
        "- COMPLETE: every autonomous documented functional intention is covered.\n"
        "- MISSING_HIGH_LEVEL_GOALS: at least one autonomous documented "
        "functional intention is absent.\n\n"

        "For every missing intention return:\n"
        "- project_description: a focused, self-contained stakeholder-style "
        "description of exactly one missing functional intention;\n"
        "- actor: the documented responsible actor;\n"
        "- source_evidence: concise evidence grounded in the complete project "
        "description;\n"
        "- rationale: why no current HLG already covers the intention.\n\n"

        "Hard rules:\n"
        "- Consider the complete documentation and the complete HLG collection.\n"
        "- Compare each proposed missing intention with every current HLG.\n"
        "- Do not propose low-level operations or implementation details.\n"
        "- Do not rewrite or delete current HLGs.\n"
        "- Do not invent functionality, actors, constraints, quality attributes, "
        "or domain facts.\n"
        "- Do not duplicate missing intentions.\n"
        "- project_description must not contain a final HLG name.\n"
        "- project_description must not mention that a goal is missing or must "
        "be added.\n"
        "- If status is COMPLETE, missing_high_level_goals must be empty.\n"
        "- If status is MISSING_HIGH_LEVEL_GOALS, the list must be non-empty.\n"
        "- Return only valid JSON without Markdown.\n\n"

        "Output structure:\n"
        "{\n"
        '  "status": "COMPLETE | MISSING_HIGH_LEVEL_GOALS",\n'
        '  "missing_high_level_goals": [\n'
        "    {\n"
        '      "project_description": "string",\n'
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
        "**Complete project description:**\n"
        f"{project_description}\n\n"
        "**Complete current high-level-goal collection:**\n"
        f"{goals_block}\n\n"
        "Check whether every autonomous documented functional intention is "
        "covered. For each missing intention, return a normal stakeholder-style "
        "project description for the top-down generator, not a final HLG and not "
        "an instruction to add a goal.\n\n"
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
            "The documentation coverage response does not respect "
            "DocumentationCoverageLLMOutput. "
            f"Response received: {raw_response}"
        ) from exc


def evaluate_documentation_coverage(
    project_description: str,
    current_high_level_goals: HighLevelGoals,
) -> DocumentationCoverageResult:
    """
    Evaluate global HLG coverage and produce normal top-down generation
    requests for genuinely missing functional intentions.
    """
    raw_response = generate_response_llama(
        _build_documentation_coverage_user_prompt(
            project_description=project_description,
            current_high_level_goals=current_high_level_goals,
        ),
        _build_documentation_coverage_system_prompt(),
    )

    output = _parse_documentation_coverage(raw_response)

    seen_intentions: set[tuple[str, str]] = set()
    retained_proposals = []
    generation_requests: list[HighLevelGoalGenerationRequest] = []

    for proposal in output.missing_high_level_goals:
        intention_key = (
            normalize_goal_name(proposal.actor.name),
            normalize_goal_name(proposal.project_description),
        )
        if intention_key in seen_intentions:
            continue

        seen_intentions.add(intention_key)
        retained_proposals.append(proposal)

        request_index = len(generation_requests) + 1
        generation_requests.append(
            HighLevelGoalGenerationRequest(
                request_id=(
                    f"documentation_coverage_add_hlg_{request_index:03d}"
                ),
                action="ADD_NEW_HIGH_LEVEL_GOAL",
                source="DOCUMENTATION_COVERAGE",
                generator_input=HighLevelGoalGeneratorInput(
                    project_description=proposal.project_description,
                    actors=Actors(actors=[proposal.actor]),
                ),
                target_branch_id=None,
                origin_branch_id=None,
                rationale=(
                    f"{proposal.rationale} Source evidence: "
                    f"{proposal.source_evidence}"
                ),
            )
        )

    if (
        output.status == "MISSING_HIGH_LEVEL_GOALS"
        and not generation_requests
    ):
        raise DocumentationCoverageEvaluationError(
            "Missing HLGs were reported, but no unique generation request "
            "could be constructed."
        )

    return DocumentationCoverageResult(
        status=output.status,
        missing_high_level_goals=retained_proposals,
        generation_requests=generation_requests,
        observations=output.observations,
    )


def save_documentation_coverage(
    output_file: str | Path,
    coverage_result: DocumentationCoverageResult,
) -> Path:
    """Persist coverage requests without applying or generating any HLG."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "3.0",
        "status": "READY_FOR_ORCHESTRATION",
        "documentation_coverage": coverage_result.model_dump(mode="json"),
        "created_at_utc": _utc_timestamp(),
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
