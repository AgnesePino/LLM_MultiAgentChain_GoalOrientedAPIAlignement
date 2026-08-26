"""Four small Llama evaluations followed by deterministic aggregation."""

import json
import re

from pydantic import BaseModel

from src.data_model import (
    BottomUpHighLevelGoal,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationResult,
    GoalBranch,
    HighLevelGoalDecision,
    HighLevelGoalEvaluation,
    HighLevelGoalReplacementRequest,
    HighLevelGoals,
    LowLevelGoalDecision,
    LowLevelGoalEvaluation,
    Actors,
    MissingHighLevelGoalEvaluation,
)
from src.llm_clients import generate_response_llama


SYSTEM_PROMPT = (
    "You are an expert in Software Engineering, Requirements Engineering, "
    "and Goal-Oriented Requirements Engineering. Return only valid JSON."
)

HLG_EVALUATOR_EXAMPLES = """Evaluation examples:
- Supported actor intention at WHY level -> KEEP_ORIGINAL_HIGH_LEVEL_GOAL.
- Generic, narrow, UI/API-oriented, or incorrectly scoped intention -> REWRITE_ORIGINAL_HIGH_LEVEL_GOAL.
- Intention unsupported by the project description -> REMOVE_ORIGINAL_HIGH_LEVEL_GOAL.
"""
LLG_EVALUATOR_EXAMPLES = """Evaluation examples:
- Atomic API-mappable LLGs covering the parent intention -> KEEP_LOW_LEVEL_GOALS.
- Missing, contradictory, or unrelated LLGs -> REGENERATE_LOW_LEVEL_GOALS.
- Technical/API wording alone is not an error when the decomposition is complete.
"""
COVERAGE_EVALUATOR_EXAMPLES = """Evaluation examples:
- Existing HLG with the same actor covers the intention -> NO_MISSING_HIGH_LEVEL_GOALS.
- Only a distinct, documented functional intention absent from current HLGs -> MISSING_HIGH_LEVEL_GOALS_FOUND.
- Do not split one intention into variants or propose technical/API goals.
"""


def _json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Evaluator response does not contain a JSON object.")
    return json.loads(cleaned[start : end + 1])


def _evaluate(prompt: str, model: type[BaseModel]):
    response = generate_response_llama(prompt, SYSTEM_PROMPT)
    try:
        return model.model_validate(_json_object(response))
    except (ValueError, TypeError, json.JSONDecodeError):
        # Keep the cycle usable when a provider returns prose or an empty
        # response. The current branch is preserved and the next iteration
        # can still perform the global coverage check.
        name = model.__name__
        if name == "HighLevelGoalEvaluation":
            return model.model_validate({
                "rationale": "Malformed evaluator response; keeping the current HLG.",
                "decision": "KEEP_ORIGINAL_HIGH_LEVEL_GOAL",
                "confidence": "LOW",
            })
        if name == "LowLevelGoalEvaluation":
            return model.model_validate({
                "rationale": "Malformed evaluator response; keeping the current LLGs.",
                "decision": "KEEP_LOW_LEVEL_GOALS",
                "confidence": "LOW",
            })
        return model.model_validate({
            "decision": "NO_MISSING_HIGH_LEVEL_GOALS",
            "rationale": "Malformed evaluator response; retaining the current HLG coverage.",
            "missing_goal_requests": [],
        })


def evaluate_original_hlg(
    project_description: str,
    branch: GoalBranch,
    reconstruction: BottomUpHighLevelGoal,
) -> HighLevelGoalEvaluation:
    prompt = f"""Evaluator rationale from the bottom-up reconstruction:
{reconstruction.rationale}

Evaluator calibration:
{HLG_EVALUATOR_EXAMPLES}

Decide whether the original High-Level Goal should be kept,
rewritten, or removed. The project description is the source of truth. Keep a
supported and reasonably scoped functional intention. Rewrite it only when it
is too generic, narrow, ambiguous, incorrectly scoped, or less faithful than
the reconstruction. Remove it only when documentation does not support it.

Complete project description:
{project_description}

Actor:
{branch.high_level_goal.actor.model_dump_json()}

Original High-Level Goal:
{branch.high_level_goal.model_dump_json()}

Bottom-up reconstructed High-Level Goal:
{reconstruction.reconstructed_high_level_goal}

Do not rewrite a supported HLG merely because its LLGs use API-like wording.

Output JSON:
{{"rationale":"...","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL | REWRITE_ORIGINAL_HIGH_LEVEL_GOAL | REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","rewriting_focus":null,"confidence":"HIGH | MEDIUM | LOW"}}"""
    return _evaluate(prompt, HighLevelGoalEvaluation)


def build_replacement_request(
    project_description: str,
    branch: GoalBranch,
    reconstruction: BottomUpHighLevelGoal,
    evaluation: HighLevelGoalEvaluation,
) -> HighLevelGoalReplacementRequest:
    prompt = f"""Evaluator rationale and rewriting focus:
{evaluation.rewriting_focus}

Prepare a focused request for the original top-down HLG
generator. Do not generate the final HLG. Describe one documented functional
intention, preserve the actor, omit pipeline internals and omit a final HLG name.

Complete project description:
{project_description}

Actor:
{branch.high_level_goal.actor.model_dump_json()}

Original HLG:
{branch.high_level_goal.model_dump_json()}

Reconstructed HLG:
{reconstruction.reconstructed_high_level_goal}

Calibration examples:
- Complete, atomic API-mappable LLGs covering the parent intention: KEEP_LOW_LEVEL_GOALS.
- Regenerate only when an important documented capability is missing, contradicted,
  or the LLGs are unrelated to the parent HLG. Technical wording alone is not an error.

Output JSON:
{{"rationale":"...","generation_project_description":"...","actor":{{"name":"...","description":"..."}}}}"""
    return _evaluate(prompt, HighLevelGoalReplacementRequest)


def evaluate_llg_decomposition(
    project_description: str,
    branch: GoalBranch,
    reconstruction: BottomUpHighLevelGoal,
) -> LowLevelGoalEvaluation:
    llgs = [goal.model_dump(mode="json") for goal in branch.low_level_goals]
    prompt = f"""Evaluator rationale and calibration:
{reconstruction.rationale}
{LLG_EVALUATOR_EXAMPLES}

Decide whether the current LLGs correctly and completely
decompose their valid parent HLG. Regenerate them if they are incomplete, too
narrow, too operational, inconsistent, or unsupported. Do not change the HLG.

Complete project description:
{project_description}

Parent HLG:
{branch.high_level_goal.model_dump_json()}

Current LLGs:
{json.dumps(llgs, ensure_ascii=False)}

Reconstructed HLG:
{reconstruction.reconstructed_high_level_goal}

Use strings (not numbers) for every item in unsupported_or_misleading_llg_ids.

Output JSON:
{{"rationale":"...","decision":"KEEP_LOW_LEVEL_GOALS | REGENERATE_LOW_LEVEL_GOALS","regeneration_feedback":null,"unsupported_or_misleading_llg_ids":[],"confidence":"HIGH | MEDIUM | LOW"}}"""
    return _evaluate(prompt, LowLevelGoalEvaluation)


def evaluate_missing_high_level_goals(
    project_description: str,
    current_hlgs: HighLevelGoals,
    actors: Actors,
) -> MissingHighLevelGoalEvaluation:
    """Check project-wide HLG coverage once per bottom-up iteration."""
    prompt = f"""Evaluator rationale and calibration:
{COVERAGE_EVALUATOR_EXAMPLES}

Check whether the project description contains functional,
actor-level High-Level Goals that are missing from the current HLG list.

Do not generate final HLG objects. Return only focused requests for the normal
top-down HLG generator. Do not propose goals already covered by current HLGs,
technical/API-level operations, or intentions unsupported by the description.

Complete project description:
{project_description}

Already identified actors:
{actors.model_dump_json()}

Current High-Level Goals:
{json.dumps([goal.model_dump(mode="json") for goal in current_hlgs.goals], ensure_ascii=False)}

Output only valid JSON:
{{
  "decision": "NO_MISSING_HIGH_LEVEL_GOALS | MISSING_HIGH_LEVEL_GOALS_FOUND",
  "rationale": "...",
  "missing_goal_requests": [
    {{"actor": "Actor name", "generation_project_description": "Focused stakeholder description"}}
  ]
}}"""
    return _evaluate(prompt, MissingHighLevelGoalEvaluation)


def evaluate_branch(
    project_description: str,
    branch: GoalBranch,
    reconstruction: BottomUpHighLevelGoal,
    current_hlgs: HighLevelGoals,
) -> GlobalGoalEvaluationResult:
    hlg = evaluate_original_hlg(project_description, branch, reconstruction)
    replacement = None
    llg = None

    if hlg.decision == HighLevelGoalDecision.REMOVE:
        decision = GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
    else:
        if hlg.decision == HighLevelGoalDecision.REWRITE:
            replacement = build_replacement_request(
                project_description, branch, reconstruction, hlg
            )

        if hlg.decision == HighLevelGoalDecision.REWRITE:
            decision = GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
        else:
            llg = evaluate_llg_decomposition(
                project_description, branch, reconstruction
            )
            decision = (
                GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS
                if llg.decision == LowLevelGoalDecision.REGENERATE
                else GlobalGoalEvaluationDecision.CONFIRM_BRANCH
            )

    rationale_parts = [hlg.rationale]
    if llg is not None:
        rationale_parts.append(llg.rationale)

    return GlobalGoalEvaluationResult(
        branch_id=branch.branch_id,
        rationale=" ".join(rationale_parts),
        final_decision=decision,
        high_level_evaluation=hlg,
        replacement_request=replacement,
        low_level_evaluation=llg,
    )


def evaluate_all_branches(
    project_description: str,
    branches: list[GoalBranch],
    reconstructions: dict[str, BottomUpHighLevelGoal],
    current_hlgs: HighLevelGoals,
):
    return {
        branch.branch_id: evaluate_branch(
            project_description,
            branch,
            reconstructions[branch.branch_id],
            current_hlgs,
        )
        for branch in branches
    }
