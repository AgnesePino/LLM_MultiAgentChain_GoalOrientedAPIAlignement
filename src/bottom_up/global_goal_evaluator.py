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
- when semantic duplicates are detected among newly generated HLGs and/or
  HLGs already present in the cycle state, it selects exactly one existing
  candidate as the representative without rewriting or merging candidates;
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
    HighLevelGoal,
    HighLevelGoals,
    LowLevelGoal,
)
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

from src.bottom_up.models import (
    BottomUpHighLevelGoal,
    BranchConsistencyLLMOutput,
    DocumentationCoverageLLMOutput,
    DocumentationCoverageResult,
    DocumentationCoverageStatus,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationLLMOutput,
    GlobalGoalEvaluationResult,
    GoalDiscoveryLLMOutput,
    HighLevelGoalGenerationAction,
    HighLevelGoalGenerationRequest,
    HighLevelGoalGeneratorInput,
    HighLevelGoalGenerationSource,
    HighLevelGoalDuplicateCandidateSource,
    HighLevelGoalDuplicateSelectionLLMOutput,
    OtherGoalMatchLLMOutput,
)
from src.bottom_up.goal_reconstructor import normalize_goal_name
from src.bottom_up.semantic_similarity import find_semantic_duplicate_high_level_goal
from src.llm_clients import generate_response_llama, MAX_SEMANTIC_RETRIES


class GlobalGoalEvaluationError(ValueError):
    """Raised when a branch-evaluation response is invalid or inconsistent."""


_OPERATIONAL_ACTION_BY_DECISION: dict[GlobalGoalEvaluationDecision, str] = {
    GlobalGoalEvaluationDecision.CONFIRM_BRANCH: "KEEP_PARENT",
    GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS: (
        "KEEP_PARENT_AND_REGENERATE_LOW_LEVEL_GOALS"
    ),
    GlobalGoalEvaluationDecision.MATCHES_OTHER_HIGH_LEVEL_GOAL: (
        "KEEP_PARENT_REGENERATE_LOW_LEVEL_GOALS_AND_RECORD_MISASSIGNMENT"
    ),
    GlobalGoalEvaluationDecision.ADD_NEW_HIGH_LEVEL_GOAL: (
        "GENERATE_NEW_HIGH_LEVEL_GOAL_AND_REGENERATE_LOW_LEVEL_GOALS"
    ),
    GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL: (
        "GENERATE_REPLACEMENT_HIGH_LEVEL_GOAL_AND_REGENERATE_LOW_LEVEL_GOALS"
    ),
}

_HLG_GENERATION_DECISIONS = {
    GlobalGoalEvaluationDecision.ADD_NEW_HIGH_LEVEL_GOAL,
    GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL,
}


_GORE_EXPERT_PREAMBLE = (
    "You are an expert in Software Engineering, Requirements Engineering, "
    "and Goal-Oriented Requirements Engineering (GORE).\n\n"
)

_GENERATION_DESCRIPTION_RULES = (
    "Rules for generation_project_description:\n"
    "- Describe exactly one autonomous functional intention.\n"
    "- Include only information supported by the complete project description.\n"
    "- Write ordinary stakeholder project documentation, not an instruction or "
    "critique.\n"
    "- Do not include a final HLG name.\n"
    "- Do not mention any branch identifier.\n"
    "- Do not mention current or previous goals, or the existing HLG "
    "collection.\n"
    "- Do not mention addition, correction, rewriting, replacement, or "
    "evaluation.\n"
    "- Do not introduce unsupported functionality, implementation details, "
    "quality attributes, or domain facts.\n\n"
)


def _build_branch_consistency_system_prompt() -> str:
    """Stage 1 system prompt: constrain the model to the single parent/candidate/decomposition-consistency question, forbidding any cross-goal comparison."""
    return (
        _GORE_EXPERT_PREAMBLE +

        "You are STAGE 1 (BRANCH CONSISTENCY) of the GLOBAL GOAL EVALUATOR. "
        "Answer exactly one question: does the current low-level decomposition "
        "reconstruct its own parent high-level goal correctly?\n\n"

        "You are given: the complete project description; the original parent "
        "high-level goal; the candidate high-level goal reconstructed "
        "bottom-up from the low-level goals belonging to the branch; and the "
        "branch's low-level goals split into ALL source, SUPPORTING (the "
        "evidence the bottom-up reconstructor selected), and NON-SUPPORTING "
        "(source goals it did not select) low-level goals. This split is the "
        "reconstructor's own interpretation, not a verified fact: inspect the "
        "low-level goals' own text rather than trusting it.\n\n"

        "The complete project description is the primary source of truth. Both "
        "the parent and the candidate may be correct, incomplete, overly "
        "generic, or unsupported.\n\n"

        "Evaluate exactly four dimensions:\n"
        "1. parent_supported: is the original parent high-level goal supported "
        "by the complete project description?\n"
        "2. candidate_supported: is the reconstructed candidate supported by "
        "the complete project description?\n"
        "3. preserves_parent_intention: check whether the candidate preserves "
        "the essential functional intention, responsible actor, relevant "
        "scope, and intended outcome of the parent goal.\n"
        "4. decomposition_complete: is the low-level decomposition complete "
        "and coherent, without unexplained non-supporting low-level goals and "
        "without missing essential responsibilities of the parent?\n\n"

        "Hard rules:\n"
        "- Use semantic meaning, not lexical similarity.\n"
        "- Do not compare the candidate with any other high-level goal here.\n"
        "- Do not decide whether the candidate matches another goal, whether a "
        "new goal must be added, or whether the parent must be rewritten: "
        "those questions belong to later stages.\n"
        "- The rationale must be concise and identify only the decisive "
        "evidence for the consistency assessment.\n"
        "- Return only valid JSON, without Markdown or additional text.\n\n"

        "Output structure:\n"
        "{\n"
        '  "rationale": "string",\n'
        '  "parent_supported": true or false,\n'
        '  "candidate_supported": true or false,\n'
        '  "preserves_parent_intention": true or false,\n'
        '  "decomposition_complete": true or false\n'
        "}\n\n"

        "Respond only with the JSON object."
    )


def _build_other_goal_match_system_prompt() -> str:
    """Stage 2 system prompt: ask only whether the candidate matches another already-extracted HLG, and enforce the branch_id-not-goal_name identifier rule."""
    return (
        _GORE_EXPERT_PREAMBLE +

        "You are STAGE 2 (OTHER-GOAL MATCH) of the GLOBAL GOAL EVALUATOR. This "
        "branch's candidate did not reconstruct its own parent. Answer exactly "
        "one question: does the candidate high-level goal reconstructed "
        "bottom-up from the low-level goals belonging to the branch correspond "
        "to another already-extracted high-level goal of the project?\n\n"

        "You are given: the candidate; the complete set of other "
        "already-extracted high-level goals of the project, each shown with "
        "its own branch_id (which is not definitive: they may themselves be "
        "incomplete, duplicated, or incorrectly scoped); the complete project "
        "description; and the branch's non-supporting low-level goals.\n\n"

        "The complete project description remains the source of truth. Judge "
        "the match semantically, not lexically.\n\n"

        "Do not decide here whether a new goal must be added or whether a "
        "parent must be rewritten: those questions belong to a later stage.\n\n"

        "IMPORTANT IDENTIFIER RULE:\n"
        "When matches_other_high_level_goal is true, matched_branch_id must "
        "be copied exactly from the branch_id shown for the matched goal.\n\n"
        "Example:\n"
        "- branch_id: branch_005\n"
        "  goal_name: HLG_005\n"
        "  description: ...\n\n"
        "Correct:\n"
        '"matched_branch_id": "branch_005"\n\n'
        "Incorrect:\n"
        '"matched_branch_id": "HLG_005"\n\n'
        "Do not return the goal_name.\n"
        "Do not return the goal's label or title.\n"
        "Do not invent, transform, shorten, or reformat the identifier.\n\n"

        "Hard rules:\n"
        "- Use semantic meaning, not lexical similarity.\n"
        "- matched_branch_id must be one of the branch_id values shown to "
        "you; it must never equal the current branch_id.\n"
        "- Compare the candidate with every listed high-level goal before "
        "concluding there is no match.\n"
        "- The rationale must be concise and identify only the decisive "
        "semantic evidence supporting or rejecting the match.\n"
        "- Return only valid JSON, without Markdown or additional text.\n\n"

        "Output structure:\n"
        "{\n"
        '  "rationale": "string",\n'
        '  "matches_other_high_level_goal": true or false,\n'
        '  "matched_branch_id": "string or null"\n'
        "}\n\n"

        "Respond only with the JSON object."
    )


def _build_goal_discovery_system_prompt() -> str:
    """Stage 3 system prompt: choose REGENERATE_LOW_LEVEL_GOALS / ADD_NEW_HIGH_LEVEL_GOAL / REWRITE_ORIGINAL_HIGH_LEVEL_GOAL once Stages 1-2 found no match."""
    return (
        _GORE_EXPERT_PREAMBLE +

        "You are STAGE 3 (GOAL DISCOVERY) of the GLOBAL GOAL EVALUATOR. This "
        "branch's candidate does not correctly reconstruct its own parent and "
        "does not correspond to another already-extracted high-level goal of "
        "the project. Answer exactly one question: what does the evidence "
        "indicate?\n\n"

        "A separate top-down component generates high-level goals. You must "
        "never generate a final high-level-goal name or description. When your "
        "outcome requires a new or replacement HLG, provide only a focused, "
        "self-contained project description in generation_project_description. "
        "That description will be passed to the normal top-down HLG generator "
        "together with the relevant actor, exactly as in an initial "
        "generation.\n\n"

        "Return exactly one outcome:\n\n"

        "- REGENERATE_LOW_LEVEL_GOALS: the parent remains correct and "
        "documented, but the low-level goals are incomplete, incoherent, too "
        "generic, too broad, semantically distorted, or insufficient, and "
        "there is no positive evidence that the parent itself is wrong and no "
        "sufficient evidence of a new autonomous intention. "
        "generation_project_description must be null.\n\n"

        "- ADD_NEW_HIGH_LEVEL_GOAL: choose this only when the evidence reveals "
        "an autonomous functional intention that does not correspond to the "
        "parent or to any other already-extracted high-level goal, is "
        "explicitly or unambiguously supported by the complete project "
        "description, belongs to the appropriate actor, and is a genuine goal "
        "rather than an operational responsibility or detail of the parent. "
        "The current parent is not replaced.\n\n"

        "- REWRITE_ORIGINAL_HIGH_LEVEL_GOAL: choose this only when the "
        "complete project description gives positive evidence that the "
        "original parent is incorrect, incomplete, ambiguous, unsupported, or "
        "incorrectly scoped, the problem is not resolvable by simply "
        "regenerating the low-level goals, and the responsible actor remains "
        "the current branch actor.\n\n"

        + _GENERATION_DESCRIPTION_RULES +

        "Hard rules:\n"
        "- Use semantic meaning, not lexical similarity.\n"
        "- When uncertain between REGENERATE_LOW_LEVEL_GOALS and one of the "
        "other outcomes, choose REGENERATE_LOW_LEVEL_GOALS.\n"
        "- Do not treat a minor operation or implementation detail as an "
        "autonomous high-level goal.\n"
        "- Do not generate or evaluate individual low-level goals.\n"
        "- The rationale must be concise and identify only the decisive "
        "semantic evidence supporting the selected outcome.\n"
        "- Return only valid JSON, without Markdown or additional text.\n\n"

        "Output structure:\n"
        "{\n"
        '  "rationale": "string",\n'
        '  "outcome": "REGENERATE_LOW_LEVEL_GOALS | ADD_NEW_HIGH_LEVEL_GOAL | '
        'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",\n'
        '  "generation_project_description": "string or null"\n'
        "}\n\n"

        "Respond only with the JSON object."
    )


def _format_low_level_goal_block(
    branch_id: str,
    full_ids: list[str],
    low_level_goals_by_id: dict[str, LowLevelGoal],
) -> str:
    """
    Render one [id] text block for a list of full low-level-goal ids.

    Every id must be resolvable in ``low_level_goals_by_id``: a missing id
    is a data-consistency bug in the branch/LLG indexing upstream, not a
    reason to silently drop that low-level goal from what the Global
    Evaluator sees. Raises GlobalGoalEvaluationError instead.
    """
    if not full_ids:
        return "(none)"

    missing_ids = [
        full_id for full_id in full_ids if full_id not in low_level_goals_by_id
    ]
    if missing_ids:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': low-level goal id(s) {sorted(missing_ids)} "
            "are missing from the branch's low-level-goal index. Available "
            f"ids are: {sorted(low_level_goals_by_id.keys())}."
        )

    return "\n".join(
        f"- [{full_id}] {low_level_goals_by_id[full_id].description}"
        for full_id in full_ids
    )


def _format_parent_block(original_high_level_goal: HighLevelGoal) -> str:
    return (
        f"- Actor: {original_high_level_goal.actor.name} - "
        f"{original_high_level_goal.actor.description}\n"
        f"- Name: {original_high_level_goal.name}\n"
        f"- Description: {original_high_level_goal.description}"
    )


def _format_candidate_block(
    reconstructed_high_level_goal: BottomUpHighLevelGoal,
) -> str:
    return (
        f"- Reconstructed goal: "
        f"{reconstructed_high_level_goal.reconstructed_high_level_goal}\n"
        f"- Abstraction rationale: "
        f"{reconstructed_high_level_goal.abstraction_rationale}\n"
        f"- Cohesion: {reconstructed_high_level_goal.cohesion}\n"
        f"- Confidence: {reconstructed_high_level_goal.confidence}"
    )


def _format_existing_high_level_goals_block(
    existing_high_level_goals: dict[str, HighLevelGoal],
) -> str:
    """
    Render each HLG with its branch_id and goal_name as clearly separate
    labeled fields, so the model cannot confuse the opaque branch identifier
    it must return (branch_id) with the HLG's own name (goal_name).
    """
    if not existing_high_level_goals:
        return "(none)"

    return "\n".join(
        f"- branch_id: {existing_branch_id}\n"
        f"  goal_name: {hlg.name}\n"
        f"  description: {hlg.description}\n"
        f"  actor: {hlg.actor.name}"
        for existing_branch_id, hlg in existing_high_level_goals.items()
    )


def _build_branch_consistency_user_prompt(
    branch_id: str,
    project_description: str,
    original_high_level_goal: HighLevelGoal,
    reconstructed_high_level_goal: BottomUpHighLevelGoal,
    low_level_goals_by_id: dict[str, LowLevelGoal],
) -> str:
    """Stage 1 user prompt: parent, reconstructed candidate, and the source/supporting/non-supporting low-level-goal blocks for one branch."""
    source_block = _format_low_level_goal_block(
        branch_id,
        reconstructed_high_level_goal.source_low_level_goal_ids,
        low_level_goals_by_id,
    )
    supporting_block = _format_low_level_goal_block(
        branch_id,
        reconstructed_high_level_goal.supporting_low_level_goal_ids,
        low_level_goals_by_id,
    )
    non_supporting_block = _format_low_level_goal_block(
        branch_id,
        reconstructed_high_level_goal.non_supporting_low_level_goal_ids,
        low_level_goals_by_id,
    )

    return (
        f"**Current branch_id:** {branch_id}\n\n"
        "**Complete project description:**\n"
        f"{project_description}\n\n"
        "**Original parent high-level goal:**\n"
        f"{_format_parent_block(original_high_level_goal)}\n\n"
        "**Candidate high-level goal reconstructed bottom-up from the "
        "low-level goals belonging to the branch:**\n"
        f"{_format_candidate_block(reconstructed_high_level_goal)}\n\n"
        "**All source low-level goals for this branch:**\n"
        f"{source_block}\n\n"
        "**Supporting low-level goals selected by the bottom-up "
        "reconstructor:**\n"
        f"{supporting_block}\n\n"
        "**Non-supporting low-level goals (source goals the reconstructor "
        "did not select as supporting evidence):**\n"
        f"{non_supporting_block}\n\n"
        "**Output:**"
    )


def _build_other_goal_match_user_prompt(
    branch_id: str,
    project_description: str,
    reconstructed_high_level_goal: BottomUpHighLevelGoal,
    existing_high_level_goals: dict[str, HighLevelGoal],
    low_level_goals_by_id: dict[str, LowLevelGoal],
) -> str:
    """Stage 2 user prompt: candidate, its non-supporting low-level goals, and every other HLG (labeled branch_id + goal_name) to match against."""
    non_supporting_block = _format_low_level_goal_block(
        branch_id,
        reconstructed_high_level_goal.non_supporting_low_level_goal_ids,
        low_level_goals_by_id,
    )

    return (
        f"**Current branch_id (excluded from a possible match):** {branch_id}\n\n"
        "**Complete project description:**\n"
        f"{project_description}\n\n"
        "**Candidate high-level goal reconstructed bottom-up from the "
        "low-level goals belonging to the branch:**\n"
        f"{_format_candidate_block(reconstructed_high_level_goal)}\n\n"
        "**Non-supporting low-level goals (source goals the reconstructor "
        "did not select as supporting evidence):**\n"
        f"{non_supporting_block}\n\n"
        "**Complete set of already-extracted high-level goals of the "
        "project (which is not definitive):**\n"
        f"{_format_existing_high_level_goals_block(existing_high_level_goals)}\n\n"
        "**Output:**"
    )


def _build_goal_discovery_user_prompt(
    branch_id: str,
    project_description: str,
    original_high_level_goal: HighLevelGoal,
    reconstructed_high_level_goal: BottomUpHighLevelGoal,
    low_level_goals_by_id: dict[str, LowLevelGoal],
) -> str:
    """Stage 3 user prompt: same branch context as Stage 1, reused now that Stages 1-2 have both ruled out a match."""
    source_block = _format_low_level_goal_block(
        branch_id,
        reconstructed_high_level_goal.source_low_level_goal_ids,
        low_level_goals_by_id,
    )
    supporting_block = _format_low_level_goal_block(
        branch_id,
        reconstructed_high_level_goal.supporting_low_level_goal_ids,
        low_level_goals_by_id,
    )
    non_supporting_block = _format_low_level_goal_block(
        branch_id,
        reconstructed_high_level_goal.non_supporting_low_level_goal_ids,
        low_level_goals_by_id,
    )

    return (
        f"**Current branch_id:** {branch_id}\n\n"
        "**Complete project description:**\n"
        f"{project_description}\n\n"
        "**Original parent high-level goal:**\n"
        f"{_format_parent_block(original_high_level_goal)}\n\n"
        "**Candidate high-level goal reconstructed bottom-up from the "
        "low-level goals belonging to the branch:**\n"
        f"{_format_candidate_block(reconstructed_high_level_goal)}\n\n"
        "**All source low-level goals for this branch:**\n"
        f"{source_block}\n\n"
        "**Supporting low-level goals selected by the bottom-up "
        "reconstructor:**\n"
        f"{supporting_block}\n\n"
        "**Non-supporting low-level goals (source goals the reconstructor "
        "did not select as supporting evidence):**\n"
        f"{non_supporting_block}\n\n"
        "This candidate has already been found not to correctly reconstruct "
        "its own parent, and not to correspond to any other already-extracted "
        "high-level goal of the project. Decide what the evidence indicates.\n\n"
        "**Output:**"
    )


def _resolve_matched_goal(
    branch_id: str,
    matched_branch_id: str | None,
    existing_high_level_goals: dict[str, HighLevelGoal],
) -> HighLevelGoal:
    """
    Look up the HLG referenced by a Stage 2 ``matched_branch_id``, raising
    ``GlobalGoalEvaluationError`` if it is missing, empty, or self-referential
    (see the hard rules in ``_build_other_goal_match_system_prompt``).
    """
    if not matched_branch_id:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': MATCHES_OTHER_HIGH_LEVEL_GOAL requires "
            "matched_branch_id."
        )

    if matched_branch_id == branch_id:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': MATCHES_OTHER_HIGH_LEVEL_GOAL cannot "
            "reference the current branch itself."
        )

    matched = existing_high_level_goals.get(matched_branch_id)
    if matched is None:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': unknown matched branch id "
            f"'{matched_branch_id}'. Valid branch ids are: "
            f"{sorted(key for key in existing_high_level_goals if key != branch_id)}."
        )

    return matched


def _build_generation_request(
    branch_id: str,
    decision: GlobalGoalEvaluationDecision,
    generation_project_description: str | None,
    rationale: str,
    original_high_level_goal: HighLevelGoal,
) -> HighLevelGoalGenerationRequest:
    """
    Convert an ADD/REWRITE decision into a request compatible with the
    original top-down HLG generator.

    Only ``generator_input`` is forwarded to the generator. Action and branch
    identifiers remain private orchestration metadata.
    """
    if decision not in _HLG_GENERATION_DECISIONS:
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': decision '{decision.value}' does not "
            "require an HLG generation request."
        )

    if not generation_project_description or not generation_project_description.strip():
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': decision '{decision.value}' requires a "
            "non-empty generation_project_description."
        )

    if decision == GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL:
        action = HighLevelGoalGenerationAction.REPLACE_EXISTING_HIGH_LEVEL_GOAL
        request_id = f"{branch_id}_replace_existing_hlg"
        target_branch_id = branch_id
    else:
        action = HighLevelGoalGenerationAction.ADD_NEW_HIGH_LEVEL_GOAL
        request_id = f"{branch_id}_add_new_hlg"
        target_branch_id = None

    return HighLevelGoalGenerationRequest(
        request_id=request_id,
        action=action,
        source=HighLevelGoalGenerationSource.BRANCH_EVALUATION,
        generator_input=HighLevelGoalGeneratorInput(
            project_description=generation_project_description,
            actors=Actors(actors=[original_high_level_goal.actor]),
        ),
        target_branch_id=target_branch_id,
        origin_branch_id=branch_id,
        rationale=rationale,
    )


def _combine_rationale(*parts: str | None) -> str:
    """Join the decisive rationale of each stage actually executed."""
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return " ".join(cleaned)


def _build_hlg_duplicate_selection_system_prompt() -> str:
    """Prompt for choosing one representative from any HLG duplicate cluster."""
    return (
        _GORE_EXPERT_PREAMBLE
        + "You are the HIGH-LEVEL GOAL DUPLICATE SELECTOR used by the "
        "bottom-up feedback loop after semantic similarity has identified a "
        "cluster of High-Level Goals that may express the same underlying "
        "intention.\n\n"
        "The cluster may contain newly GENERATED HLGs, HLGs that already "
        "EXIST in the current cycle state, or both. Candidate origin is "
        "traceability information only: never prefer an EXISTING candidate "
        "just because it already exists, and never prefer a GENERATED "
        "candidate just because it is newer.\n\n"
        "Your task is ONLY to choose which supplied candidate should survive "
        "as the single representative of the duplicated intention. The "
        "complete project description is the primary source of truth. The "
        "focused generation request identifies the particular documented "
        "intention that triggered this comparison.\n\n"
        "Do NOT create a new goal. Do NOT rewrite, merge, combine, shorten, "
        "or improve candidates. Select exactly one candidate identifier from "
        "those supplied by the user. Every other candidate in this duplicate "
        "cluster will be removed by deterministic Python logic.\n\n"
        "Selection criteria, in order of importance:\n"
        "1. PROJECT-DOCUMENTATION FAITHFULNESS: the candidate must be directly "
        "supported by the complete project description and must preserve the "
        "documented stakeholder intention, relevant scope, and intended "
        "outcome.\n"
        "2. FOCUSED-INTENTION COVERAGE: among supported candidates, prefer the "
        "one that most completely and precisely represents the focused "
        "project description that triggered the generation request.\n"
        "3. CORRECT GORE HIGH-LEVEL ABSTRACTION: a High-Level Goal expresses "
        "WHY / an objective or desired outcome. It must be more abstract than "
        "operational steps and must not merely describe an implementation "
        "mechanism, UI element, atomic action, data-access step, notification "
        "event, or technical procedure.\n"
        "4. AUTONOMOUS GOAL SCOPE: the candidate should express one coherent "
        "functional intention that is meaningful as a goal branch. Avoid both "
        "overly narrow operational details and vague umbrella goals that lose "
        "the documented intention.\n"
        "5. ACTOR COHERENCE: the responsible actor and stakeholder perspective "
        "must be consistent with the project documentation and the request.\n"
        "6. NO UNSUPPORTED CONTENT: reject candidates that introduce "
        "functions, outcomes, constraints, technologies, quality attributes, "
        "or domain facts not justified by the project description.\n"
        "7. CLARITY AND COHESION: if candidates remain otherwise equivalent, "
        "prefer the clearest, most cohesive formulation of the documented "
        "high-level intention.\n\n"
        "Hard rules:\n"
        "- Evaluate semantic meaning, not lexical similarity or candidate "
        "name style.\n"
        "- Do not reward a candidate merely for being broader or more "
        "generic.\n"
        "- Do not infer that an existing branch is correct merely because it "
        "was previously accepted.\n"
        "- Do not infer that a newly generated candidate is better merely "
        "because it was generated from the latest request.\n"
        "- Return only valid JSON, without Markdown or additional text.\n\n"
        "Return JSON only, with exactly these fields:\n"
        '{"selected_candidate_id": "candidate_XXX", '
        '"rationale": "brief justification grounded in the project '
        'description and HLG abstraction criteria"}'
    )


def _build_hlg_duplicate_selection_user_prompt(
    request: HighLevelGoalGenerationRequest,
    candidates: dict[str, HighLevelGoal],
    project_description: str,
    candidate_sources: dict[str, HighLevelGoalDuplicateCandidateSource],
    candidate_branch_ids: dict[str, str | None],
) -> str:
    """Build the project-grounded prompt for duplicate selection."""
    actors_block = "\n".join(
        f"- {actor.name}: {actor.description}"
        for actor in request.generator_input.actors.actors
    )

    candidate_blocks: list[str] = []
    for candidate_id, goal in candidates.items():
        source = candidate_sources[candidate_id]
        branch_id = candidate_branch_ids[candidate_id]
        branch_label = branch_id if branch_id is not None else "(not assigned)"
        candidate_blocks.append(
            "\n".join(
                [
                    f"Candidate id: {candidate_id}",
                    f"Source: {source.value}",
                    f"Current branch_id: {branch_label}",
                    f"Name: {goal.name}",
                    f"Description: {goal.description}",
                    f"Actor: {goal.actor.name} - {goal.actor.description}",
                ]
            )
        )
    candidates_block = "\n\n".join(candidate_blocks)

    return (
        "Semantic similarity has identified the candidates below as one "
        "duplicate cluster. Some candidates may already exist in the current "
        "goal hierarchy and some may have just been generated. Choose the "
        "single candidate that best represents the documented intention as a "
        "GORE High-Level Goal.\n\n"
        "**Complete project description (PRIMARY SOURCE OF TRUTH):**\n"
        f"{project_description}\n\n"
        f"Request id: {request.request_id}\n"
        f"Request source: {request.source.value}\n"
        f"Request action: {request.action.value}\n\n"
        "**Focused project description / intention that triggered this "
        "generation request:**\n"
        f"{request.generator_input.project_description}\n\n"
        "**Actor(s) supplied to the normal top-down generator:**\n"
        f"{actors_block}\n\n"
        "**Evaluator rationale that produced the generation request:**\n"
        f"{request.rationale}\n\n"
        "**Duplicate HLG candidates:**\n"
        f"{candidates_block}\n\n"
        "Candidate Source and Current branch_id are traceability metadata. "
        "They must not influence the choice except to identify the objects. "
        "Select the candidate whose NAME + DESCRIPTION + ACTOR best satisfy "
        "the complete project documentation, the focused intention, and the "
        "GORE High-Level Goal abstraction criteria. Select exactly one "
        "candidate id and do not propose any new wording."
    )


def _validate_duplicate_candidate_metadata(
    candidates: dict[str, HighLevelGoal],
    candidate_sources: dict[str, HighLevelGoalDuplicateCandidateSource],
    candidate_branch_ids: dict[str, str | None],
) -> None:
    """Validate selector metadata before exposing the cluster to the LLM."""
    candidate_ids = set(candidates)
    if set(candidate_sources) != candidate_ids:
        raise ValueError(
            "candidate_sources must contain exactly one entry for every "
            "duplicate candidate id."
        )
    if set(candidate_branch_ids) != candidate_ids:
        raise ValueError(
            "candidate_branch_ids must contain exactly one entry for every "
            "duplicate candidate id."
        )

    for candidate_id in candidates:
        source = candidate_sources[candidate_id]
        branch_id = candidate_branch_ids[candidate_id]
        if source == HighLevelGoalDuplicateCandidateSource.EXISTING:
            if branch_id is None or not branch_id.strip():
                raise ValueError(
                    f"Existing duplicate candidate '{candidate_id}' requires "
                    "a non-empty branch id."
                )
        elif branch_id is not None:
            raise ValueError(
                f"Generated duplicate candidate '{candidate_id}' must not "
                "have an existing branch id."
            )


def select_best_duplicate_high_level_goal(
    request: HighLevelGoalGenerationRequest,
    candidates: dict[str, HighLevelGoal],
    *,
    project_description: str | None = None,
    candidate_sources: (
        dict[str, HighLevelGoalDuplicateCandidateSource] | None
    ) = None,
    candidate_branch_ids: dict[str, str | None] | None = None,
) -> tuple[str, str]:
    """
    Select the single best HLG from a semantic-duplicate cluster.

    The candidates may be newly generated HLGs, already-existing HLGs, or a
    mixture of both. Selection is grounded in the complete project
    description and in explicit GORE High-Level Goal criteria. Candidate
    origin is never a preference signal.

    This remains an evaluator operation: no HLG is generated, rewritten, or
    merged here. The returned identifier always names one of the supplied
    candidates; deterministic update logic decides which losing objects must
    later be removed.

    ``project_description`` and candidate metadata are optional only for
    backward compatibility with the previous intra-generation selector. The
    updated goal-update/orchestration path should provide all three.
    """
    if len(candidates) < 2:
        raise ValueError(
            "Duplicate selection requires at least two HLG candidates."
        )

    candidate_ids = set(candidates)

    if project_description is None:
        project_description = request.generator_input.project_description
    if not project_description.strip():
        raise ValueError(
            "Duplicate selection requires a non-empty project description."
        )

    if candidate_sources is None:
        candidate_sources = {
            candidate_id: HighLevelGoalDuplicateCandidateSource.GENERATED
            for candidate_id in candidates
        }
    if candidate_branch_ids is None:
        candidate_branch_ids = {
            candidate_id: None
            for candidate_id in candidates
        }

    _validate_duplicate_candidate_metadata(
        candidates=candidates,
        candidate_sources=candidate_sources,
        candidate_branch_ids=candidate_branch_ids,
    )

    def _validate_selection(
        output: HighLevelGoalDuplicateSelectionLLMOutput,
    ) -> tuple[str, str]:
        if output.selected_candidate_id not in candidate_ids:
            raise GlobalGoalEvaluationError(
                "selected_candidate_id must be one of the supplied candidate "
                f"ids {sorted(candidate_ids)}, received "
                f"'{output.selected_candidate_id}'."
            )
        return output.selected_candidate_id, output.rationale

    return _run_evaluation_stage(
        user_prompt=_build_hlg_duplicate_selection_user_prompt(
            request=request,
            candidates=candidates,
            project_description=project_description,
            candidate_sources=candidate_sources,
            candidate_branch_ids=candidate_branch_ids,
        ),
        system_prompt=_build_hlg_duplicate_selection_system_prompt(),
        output_model=HighLevelGoalDuplicateSelectionLLMOutput,
        stage_label="high-level goal duplicate selection",
        process=_validate_selection,
    )


_StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)
_StageResultT = TypeVar("_StageResultT")


def _parse_structured_llm_output(
    raw_response: str,
    output_model: type[_StructuredOutputT],
    stage_label: str,
) -> _StructuredOutputT:
    """Parse and validate the JSON returned by one evaluator LLM stage."""
    if not raw_response or not raw_response.strip():
        raise GlobalGoalEvaluationError(
            f"Llama returned an empty {stage_label} response."
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
            f"Llama did not return valid JSON for the {stage_label} stage. "
            f"Response received: {raw_response}"
        ) from exc

    try:
        return output_model.model_validate(parsed_json)
    except PydanticValidationError as exc:
        raise GlobalGoalEvaluationError(
            f"Llama returned JSON that does not respect {output_model.__name__} "
            f"for the {stage_label} stage. Response received: {raw_response}"
        ) from exc


def _run_evaluation_stage(
    user_prompt: str,
    system_prompt: str,
    output_model: type[_StructuredOutputT],
    stage_label: str,
    process: Callable[[_StructuredOutputT], _StageResultT],
) -> _StageResultT:
    """
    Call the LLM for one evaluator stage, parse and Pydantic-validate its
    output, then run ``process`` for any further semantic check (e.g.
    resolving a referenced id). ``process`` may raise
    GlobalGoalEvaluationError, in which case the model is retried with
    feedback, up to MAX_SEMANTIC_RETRIES times, matching the retry mechanism
    already used across the evaluator.
    """
    retry_feedback = ""
    last_error: GlobalGoalEvaluationError | None = None

    for _ in range(MAX_SEMANTIC_RETRIES + 1):
        prompt = (
            user_prompt if not retry_feedback
            else f"{user_prompt}\n\n{retry_feedback}"
        )

        raw_output = generate_response_llama(prompt, system_prompt)

        try:
            parsed_output = _parse_structured_llm_output(
                raw_output, output_model, stage_label
            )
            return process(parsed_output)
        except GlobalGoalEvaluationError as exc:
            last_error = exc
            retry_feedback = (
                f"The previous answer was invalid because: {exc} "
                "Regenerate the output respecting this constraint."
            )

    raise last_error


def evaluate_branch(
    branch_id: str,
    project_description: str,
    original_high_level_goal: HighLevelGoal,
    reconstructed_high_level_goal: BottomUpHighLevelGoal,
    existing_high_level_goals: dict[str, HighLevelGoal],
    low_level_goals_by_id: dict[str, LowLevelGoal],
) -> GlobalGoalEvaluationResult:
    """
    Evaluate one branch as a conditional sequence of up to three specialized
    LLM stages, without generating or applying a final HLG:

    1. Branch consistency (parent vs. candidate vs. decomposition). If this
       confirms the branch, return CONFIRM_BRANCH immediately.
    2. Otherwise, other-goal match against the existing HLG collection. If a
       match is found, return MATCHES_OTHER_HIGH_LEVEL_GOAL immediately.
    3. Otherwise, goal discovery: REGENERATE_LOW_LEVEL_GOALS,
       ADD_NEW_HIGH_LEVEL_GOAL, or REWRITE_ORIGINAL_HIGH_LEVEL_GOAL.
    """
    stage1_output = _run_evaluation_stage(
        user_prompt=_build_branch_consistency_user_prompt(
            branch_id=branch_id,
            project_description=project_description,
            original_high_level_goal=original_high_level_goal,
            reconstructed_high_level_goal=reconstructed_high_level_goal,
            low_level_goals_by_id=low_level_goals_by_id,
        ),
        system_prompt=_build_branch_consistency_system_prompt(),
        output_model=BranchConsistencyLLMOutput,
        stage_label="branch consistency",
        process=lambda output: output,
    )

    if stage1_output.confirms_branch:
        return GlobalGoalEvaluationResult(
            branch_id=branch_id,
            decision=GlobalGoalEvaluationDecision.CONFIRM_BRANCH,
            original_high_level_goal=original_high_level_goal,
            reconstructed_high_level_goal=(
                reconstructed_high_level_goal.reconstructed_high_level_goal
            ),
            matched_high_level_goal=None,
            generation_request=None,
            rationale=stage1_output.rationale,
            operational_action=_OPERATIONAL_ACTION_BY_DECISION[
                GlobalGoalEvaluationDecision.CONFIRM_BRANCH
            ],
            requires_high_level_regeneration=False,
            requires_low_level_regeneration=False,
        )

    # Stage 2 must never be able to match the branch to itself: the current
    # branch is excluded from the candidate pool before the prompt is even
    # built, instead of being shown to the model and relying on an
    # instruction not to pick it.
    other_high_level_goals = {
        candidate_branch_id: goal
        for candidate_branch_id, goal in existing_high_level_goals.items()
        if candidate_branch_id != branch_id
    }

    if other_high_level_goals:
        def _process_stage2(
            output: OtherGoalMatchLLMOutput,
        ) -> tuple[OtherGoalMatchLLMOutput, HighLevelGoal | None]:
            if not output.matches_other_high_level_goal:
                return output, None
            matched = _resolve_matched_goal(
                branch_id=branch_id,
                matched_branch_id=output.matched_branch_id,
                existing_high_level_goals=existing_high_level_goals,
            )
            return output, matched

        stage2_output, matched_high_level_goal = _run_evaluation_stage(
            user_prompt=_build_other_goal_match_user_prompt(
                branch_id=branch_id,
                project_description=project_description,
                reconstructed_high_level_goal=reconstructed_high_level_goal,
                existing_high_level_goals=other_high_level_goals,
                low_level_goals_by_id=low_level_goals_by_id,
            ),
            system_prompt=_build_other_goal_match_system_prompt(),
            output_model=OtherGoalMatchLLMOutput,
            stage_label="other-goal match",
            process=_process_stage2,
        )
        stage2_rationale = stage2_output.rationale
        stage2_matches = stage2_output.matches_other_high_level_goal
    else:
        # No other HLG exists to compare against: the answer is
        # deterministically "no match", without spending an LLM call on a
        # question that has only one possible answer.
        matched_high_level_goal = None
        stage2_rationale = None
        stage2_matches = False

    if stage2_matches:
        return GlobalGoalEvaluationResult(
            branch_id=branch_id,
            decision=GlobalGoalEvaluationDecision.MATCHES_OTHER_HIGH_LEVEL_GOAL,
            original_high_level_goal=original_high_level_goal,
            reconstructed_high_level_goal=(
                reconstructed_high_level_goal.reconstructed_high_level_goal
            ),
            matched_high_level_goal=matched_high_level_goal,
            generation_request=None,
            rationale=_combine_rationale(
                stage1_output.rationale, stage2_rationale
            ),
            operational_action=_OPERATIONAL_ACTION_BY_DECISION[
                GlobalGoalEvaluationDecision.MATCHES_OTHER_HIGH_LEVEL_GOAL
            ],
            requires_high_level_regeneration=False,
            requires_low_level_regeneration=True,
        )

    def _process_stage3(
        output: GoalDiscoveryLLMOutput,
    ) -> tuple[
        GoalDiscoveryLLMOutput,
        GlobalGoalEvaluationDecision,
        HighLevelGoalGenerationRequest | None,
    ]:
        # GoalDiscoveryOutcome and GlobalGoalEvaluationDecision intentionally
        # share the same string values for these three outcomes, so this
        # conversion is always valid; see the enums' definitions in models.py.
        decision = GlobalGoalEvaluationDecision(output.outcome.value)
        if decision not in _HLG_GENERATION_DECISIONS:
            return output, decision, None
        generation_request = _build_generation_request(
            branch_id=branch_id,
            decision=decision,
            generation_project_description=output.generation_project_description,
            rationale=output.rationale,
            original_high_level_goal=original_high_level_goal,
        )
        return output, decision, generation_request

    stage3_output, decision, generation_request = _run_evaluation_stage(
        user_prompt=_build_goal_discovery_user_prompt(
            branch_id=branch_id,
            project_description=project_description,
            original_high_level_goal=original_high_level_goal,
            reconstructed_high_level_goal=reconstructed_high_level_goal,
            low_level_goals_by_id=low_level_goals_by_id,
        ),
        system_prompt=_build_goal_discovery_system_prompt(),
        output_model=GoalDiscoveryLLMOutput,
        stage_label="goal discovery",
        process=_process_stage3,
    )

    return GlobalGoalEvaluationResult(
        branch_id=branch_id,
        decision=decision,
        original_high_level_goal=original_high_level_goal,
        reconstructed_high_level_goal=(
            reconstructed_high_level_goal.reconstructed_high_level_goal
        ),
        matched_high_level_goal=None,
        generation_request=generation_request,
        rationale=_combine_rationale(
            stage1_output.rationale, stage2_rationale, stage3_output.rationale
        ),
        operational_action=_OPERATIONAL_ACTION_BY_DECISION[decision],
        requires_high_level_regeneration=decision in _HLG_GENERATION_DECISIONS,
        requires_low_level_regeneration=True,
    )


def evaluate_all_branches(
    project_description: str,
    existing_high_level_goals: dict[str, HighLevelGoal],
    reconstructed_high_level_goals: dict[str, BottomUpHighLevelGoal],
    empty_branches: list[str],
    branch_low_level_goals: dict[str, dict[str, LowLevelGoal]] | None = None,
) -> tuple[dict[str, GlobalGoalEvaluationResult], dict[str, str]]:
    """
    Produce exactly one validated evaluation for every expected branch.

    ``branch_low_level_goals`` maps each non-empty branch_id to
    {full_low_level_goal_id: LowLevelGoal}, so the Global Evaluator can see
    the text of the source/supporting/non-supporting low-level goals, not
    just their ids. It is optional and defaults to an empty mapping per
    branch for backward compatibility with callers that do not build it.
    """
    results: dict[str, GlobalGoalEvaluationResult] = {}
    errors: dict[str, str] = {}
    branch_low_level_goals = branch_low_level_goals or {}

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
                    low_level_goals_by_id=branch_low_level_goals.get(
                        branch_id, {}
                    ),
                )
            except Exception as exc:
                errors[branch_id] = f"{type(exc).__name__}: {exc}"
            continue

        if branch_id in empty_branch_ids:
            try:
                results[branch_id] = evaluate_empty_branch(
                    branch_id=branch_id,
                    project_description=project_description,
                    original_high_level_goal=original,
                )
            except Exception as exc:
                errors[branch_id] = f"{type(exc).__name__}: {exc}"
            continue

        errors[branch_id] = (
            "GlobalGoalEvaluationError: the expected branch has neither "
            "a reconstructed HLG nor an empty-branch marker. This indicates "
            "a missing or failed bottom-up reconstruction."
        )

    return results, errors


_EMPTY_BRANCH_ALLOWED_DECISIONS = {
    GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS,
    GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL,
}


def _build_empty_branch_system_prompt() -> str:
    """Single-stage system prompt for a branch with no low-level goals: decide REGENERATE_LOW_LEVEL_GOALS vs. REWRITE_ORIGINAL_HIGH_LEVEL_GOAL from the parent alone."""
    return (
        "You are an expert in Software Engineering, Requirements Engineering, "
        "and Goal-Oriented Requirements Engineering (GORE).\n\n"

        "You are the GLOBAL GOAL EVALUATOR evaluating an EMPTY BRANCH: a "
        "high-level goal that currently has no low-level goals, so no "
        "bottom-up candidate could be reconstructed for it.\n\n"

        "A separate top-down component generates high-level goals. You must "
        "never generate a final high-level-goal name or description. When your "
        "decision requires a replacement HLG, provide only a focused, "
        "self-contained project description in generation_project_description. "
        "That description will be passed to the normal top-down HLG generator "
        "together with the relevant actor, exactly as in an initial generation.\n\n"

        "You must compare:\n"
        "- the original high-level goal;\n"
        "- its responsible actor;\n"
        "- the complete project description.\n\n"

        "The complete project description is the primary source of truth. Read "
        "and consider the entire description before deciding.\n\n"

        "Return exactly one decision:\n\n"

        "- REGENERATE_LOW_LEVEL_GOALS: choose this when the original high-level "
        "goal is supported by the complete project description, correctly "
        "scoped, and correctly attributed to its actor. Preserve the goal and "
        "request only a new low-level decomposition. "
        "generation_project_description must be null.\n\n"

        "- REWRITE_ORIGINAL_HIGH_LEVEL_GOAL: choose this only when the complete "
        "documentation positively demonstrates that the original high-level "
        "goal is incorrect, incomplete, ambiguous, unsupported, or incorrectly "
        "scoped, while the responsible actor remains the current actor. "
        "generation_project_description must contain a focused, self-contained "
        "stakeholder description of the correct documented functional intention. "
        "Do not mention the old goal, its defects, rewriting, replacement, or "
        "evaluation.\n\n"

        "Rules for generation_project_description:\n"
        "- Use it only for REWRITE_ORIGINAL_HIGH_LEVEL_GOAL.\n"
        "- Describe exactly one autonomous functional intention.\n"
        "- Include only information supported by the complete project description.\n"
        "- Preserve the essential functional intention, responsible actor, "
        "relevant scope, and intended outcome of the original goal.\n"
        "- Write ordinary project documentation, not an instruction or critique.\n"
        "- Do not include a final HLG name.\n"
        "- Do not mention any branch identifier.\n"
        "- Do not mention current or previous goals.\n"
        "- Do not mention addition, correction, rewriting, or replacement.\n"
        "- Do not introduce unsupported functionality, implementation details, "
        "quality attributes, or domain facts.\n\n"

        "Hard rules:\n"
        "- Use semantic meaning, not lexical similarity.\n"
        "- When uncertain between REGENERATE_LOW_LEVEL_GOALS and "
        "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL, choose REGENERATE_LOW_LEVEL_GOALS.\n"
        "- Choose REWRITE_ORIGINAL_HIGH_LEVEL_GOAL only when the documentation "
        "provides positive evidence that the original goal itself is defective.\n"
        "- The rationale must be concise and identify only the decisive "
        "evidence for whether the original goal is supported by the "
        "documentation.\n"
        "- Return only valid JSON, without Markdown or additional text.\n\n"

        "Output structure:\n"
        "{\n"
        '  "rationale": "string",\n'
        '  "decision": "REGENERATE_LOW_LEVEL_GOALS | '
        'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL",\n'
        '  "generation_project_description": "string or null"\n'
        "}\n\n"

        "Respond only with the JSON object."
    )


def _build_empty_branch_user_prompt(
    branch_id: str,
    project_description: str,
    original_high_level_goal: HighLevelGoal,
) -> str:
    """Empty-branch user prompt: the parent HLG and project description only, no reconstructed candidate to compare against."""
    return (
        f"**Current branch_id:** {branch_id}\n\n"
        "**Complete project description:**\n"
        f"{project_description}\n\n"
        "**Original high-level goal (empty branch, no low-level goals):**\n"
        f"- Actor: {original_high_level_goal.actor.name} - "
        f"{original_high_level_goal.actor.description}\n"
        f"- Name: {original_high_level_goal.name}\n"
        f"- Description: {original_high_level_goal.description}\n\n"
        "Evaluate whether this original high-level goal is supported by the "
        "complete project description. When a replacement is required, return "
        "a focused stakeholder-style project description for the normal "
        "top-down generator, not a final goal and not a correction "
        "instruction.\n\n"
        "**Output:**"
    )


def evaluate_empty_branch(
    branch_id: str,
    project_description: str,
    original_high_level_goal: HighLevelGoal,
) -> GlobalGoalEvaluationResult:
    """
    Evaluate an HLG branch with no current low-level goals.

    No bottom-up candidate exists for this branch, so the decision is based
    only on the original high-level goal, its actor, and the complete project
    description, instead of the reconstructed-candidate comparison used by
    ``evaluate_branch``.
    """
    def _process_empty_branch(
        llm_output: GlobalGoalEvaluationLLMOutput,
    ) -> GlobalGoalEvaluationResult:
        decision = llm_output.decision

        if decision not in _EMPTY_BRANCH_ALLOWED_DECISIONS:
            raise GlobalGoalEvaluationError(
                f"Branch '{branch_id}': empty-branch evaluation returned "
                f"unsupported decision '{decision.value}'."
            )

        generation_request: HighLevelGoalGenerationRequest | None = None
        if decision == GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL:
            generation_request = _build_generation_request(
                branch_id=branch_id,
                decision=decision,
                generation_project_description=(
                    llm_output.generation_project_description
                ),
                rationale=llm_output.rationale,
                original_high_level_goal=original_high_level_goal,
            )

        return GlobalGoalEvaluationResult(
            branch_id=branch_id,
            decision=decision,
            original_high_level_goal=original_high_level_goal,
            reconstructed_high_level_goal=None,
            matched_high_level_goal=None,
            generation_request=generation_request,
            rationale=llm_output.rationale,
            operational_action=_OPERATIONAL_ACTION_BY_DECISION[decision],
            requires_high_level_regeneration=(
                decision
                == GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
            ),
            requires_low_level_regeneration=True,
        )

    return _run_evaluation_stage(
        user_prompt=_build_empty_branch_user_prompt(
            branch_id=branch_id,
            project_description=project_description,
            original_high_level_goal=original_high_level_goal,
        ),
        system_prompt=_build_empty_branch_system_prompt(),
        output_model=GlobalGoalEvaluationLLMOutput,
        stage_label="empty-branch evaluation",
        process=_process_empty_branch,
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
    """Project-wide coverage-check system prompt: find documented intentions not represented by any current HLG, without generating or rewriting HLGs."""
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

        "Determine whether each autonomous documented functional intention is "
        "already semantically covered by at least one current HLG.\n\n"

        "Do not require an HLG to explicitly mention every operation, channel, "
        "visualization, statistic, field, or sub-capability. If a documented "
        "functionality is semantically subsumed by the intention of an "
        "existing HLG, it is already covered and must not be proposed as "
        "missing.\n\n"

        "Before proposing a missing intention, compare it against every "
        "current HLG. Propose it only if no current HLG semantically covers "
        "that intention.\n\n"

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
        "- Compare each proposed missing intention with every current HLG "
        "before proposing it.\n"
        "- Do not propose an intention already semantically subsumed by an "
        "existing HLG, even if that HLG does not name it explicitly.\n"
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
    """Coverage user prompt: complete project description plus the full current HLG collection to check it against."""
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
    except PydanticValidationError as exc:
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
    base_user_prompt = _build_documentation_coverage_user_prompt(
        project_description=project_description,
        current_high_level_goals=current_high_level_goals,
    )
    system_prompt = _build_documentation_coverage_system_prompt()

    # Semantic retry: the response can be valid JSON while still being
    # semantically unusable (e.g. MISSING_HIGH_LEVEL_GOALS reported but every
    # proposal collapses into a duplicate intention). Give the model another
    # chance with feedback instead of failing the whole coverage check.
    retry_feedback = ""
    last_error: DocumentationCoverageEvaluationError | None = None

    for _ in range(MAX_SEMANTIC_RETRIES + 1):
        user_prompt = (
            base_user_prompt
            if not retry_feedback
            else f"{base_user_prompt}\n\n{retry_feedback}"
        )

        raw_response = generate_response_llama(user_prompt, system_prompt)

        try:
            output = _parse_documentation_coverage(raw_response)

            seen_intentions: set[tuple[str, str]] = set()
            retained_proposals = []
            generation_requests: list[HighLevelGoalGenerationRequest] = []
            coverage_observations = list(output.observations)

            for proposal in output.missing_high_level_goals:
                intention_key = (
                    normalize_goal_name(proposal.actor.name),
                    normalize_goal_name(proposal.project_description),
                )
                if intention_key in seen_intentions:
                    continue
                seen_intentions.add(intention_key)

                # Semantic pre-validation: a proposal that a current HLG of
                # the same actor already covers, in substance, must not
                # become a generation request even though it is not a
                # lexical duplicate of another proposal in this response.
                # This is the primary defense against a falsely "missing"
                # intention; goal_update.py's post-generation check remains
                # only as a final safety net.
                matched_existing_goal, similarity = (
                    find_semantic_duplicate_high_level_goal(
                        HighLevelGoal(
                            name=proposal.project_description,
                            description=proposal.project_description,
                            actor=proposal.actor,
                        ),
                        current_high_level_goals.goals,
                    )
                )
                if matched_existing_goal is not None:
                    coverage_observations.append(
                        f"Skipped proposal for actor '{proposal.actor.name}': "
                        "already semantically covered by existing HLG "
                        f"'{matched_existing_goal.name}' "
                        f"(similarity={similarity:.4f})."
                    )
                    continue

                retained_proposals.append(proposal)

                request_index = len(generation_requests) + 1
                generation_requests.append(
                    HighLevelGoalGenerationRequest(
                        request_id=(
                            f"documentation_coverage_add_hlg_{request_index:03d}"
                        ),
                        action=HighLevelGoalGenerationAction.ADD_NEW_HIGH_LEVEL_GOAL,
                        source=HighLevelGoalGenerationSource.DOCUMENTATION_COVERAGE,
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
                output.status
                == DocumentationCoverageStatus.MISSING_HIGH_LEVEL_GOALS
                and not generation_requests
            ):
                raise DocumentationCoverageEvaluationError(
                    "Missing HLGs were reported, but every proposal was "
                    "either a duplicate or already semantically covered by "
                    "an existing HLG; no unique generation request could be "
                    "constructed."
                )

            return DocumentationCoverageResult(
                status=output.status,
                missing_high_level_goals=retained_proposals,
                generation_requests=generation_requests,
                observations=coverage_observations,
            )
        except DocumentationCoverageEvaluationError as exc:
            last_error = exc
            retry_feedback = (
                f"The previous answer was invalid because: {exc} "
                "Regenerate the output respecting this constraint."
            )

    raise last_error


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
