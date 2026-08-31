"""Sequential branch and global bottom-up evaluations with deterministic aggregation."""

import json
import re

from pydantic import BaseModel

from src.data_model import (
    BottomUpHighLevelGoal,
    ConfidenceLevel,
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


def _key(value: str) -> str:
    return " ".join(value.casefold().split())


def _stable_hlg_key(branch: GoalBranch) -> str:
    return (
        f"{_key(branch.high_level_goal.actor.name)}::"
        f"{_key(branch.high_level_goal.name)}"
    )

HLG_EVALUATOR_EXAMPLES = """Evaluation examples:
- Supported actor intention at WHY level -> KEEP_ORIGINAL_HIGH_LEVEL_GOAL.
- Generic, narrow, UI/API-oriented, or incorrectly scoped intention -> REWRITE_ORIGINAL_HIGH_LEVEL_GOAL.
- Intention unsupported by the project description -> REMOVE_ORIGINAL_HIGH_LEVEL_GOAL.
"""
LLG_EVALUATOR_EXAMPLES = """Evaluation examples:
- Atomic API-mappable LLGs covering the parent intention -> KEEP_LOW_LEVEL_GOALS.
- A material capability explicitly required by the parent HLG is absent -> REGENERATE_LOW_LEVEL_GOALS.
- A current LLG is contradicted by or unsupported in the documentation -> REGENERATE_LOW_LEVEL_GOALS.
- Technical/API wording alone is not an error when the decomposition is complete.
- Different granularity, optional CRUD operations, UI steps, and merely possible
  improvements are not material defects -> KEEP_LOW_LEVEL_GOALS.
- Capabilities owned by sibling HLGs must not be copied into this branch.
"""
COVERAGE_EVALUATOR_EXAMPLES = """Evaluation examples:
- Existing HLG with the same actor covers the intention -> NO_MISSING_HIGH_LEVEL_GOALS.
- Only a distinct, documented functional intention absent from current HLGs -> MISSING_HIGH_LEVEL_GOALS_FOUND.
- Do not split one intention into variants or propose technical/API goals.
- A workflow step, CRUD operation, UI action, notification, or narrower variant
  of a current goal is already covered -> NO_MISSING_HIGH_LEVEL_GOALS.
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
        if name == "HighLevelGoalReplacementRequest":
            raise ValueError("Malformed HighLevelGoalReplacementRequest response.")
        if name == "MissingHighLevelGoalEvaluation":
            # A malformed project-wide coverage response must not abort the
            # dataset.  Conservatively retain the current HLG set and retry
            # coverage on the next iteration.
            return model.model_validate({
                "decision": "NO_MISSING_HIGH_LEVEL_GOALS",
                "rationale": (
                    "Malformed evaluator response; retaining the current HLG coverage."
                ),
                "missing_goal_requests": [],
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
    current_hlgs: HighLevelGoals | None = None,
) -> HighLevelGoalEvaluation:
    current_hlgs = current_hlgs or HighLevelGoals(goals=[branch.high_level_goal])
    branch_key = (
        _key(branch.high_level_goal.actor.name),
        _key(branch.high_level_goal.name),
    )
    sibling_hlgs = [
        goal.model_dump(mode="json")
        for goal in current_hlgs.goals
        if (_key(goal.actor.name), _key(goal.name))
        != branch_key
    ]
    prompt = f"""Evaluator rationale from the bottom-up reconstruction:
{reconstruction.rationale}

Evaluator calibration:
{HLG_EVALUATOR_EXAMPLES}

Decide whether the original High-Level Goal should be kept,
rewritten, or removed. The project description is the source of truth. Keep a
supported and reasonably scoped functional intention. Rewrite it only when it
is too generic, narrow, ambiguous, incorrectly scoped, or less faithful than
the reconstruction. Remove it when documentation does not support it or when
another HLG for the same actor already covers the same functional intention.
Judge semantic overlap, not only equal names. Do not keep several HLGs that
merely split one stakeholder intention into wording variants.

Complete project description:
{project_description}

Actor:
{branch.high_level_goal.actor.model_dump_json()}

Original High-Level Goal:
{branch.high_level_goal.model_dump_json()}

Bottom-up reconstructed High-Level Goal:
{reconstruction.reconstructed_high_level_goal}

Other current High-Level Goals:
{json.dumps(sibling_hlgs, ensure_ascii=False)}

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
    current_hlgs: HighLevelGoals | None = None,
) -> LowLevelGoalEvaluation:
    current_hlgs = current_hlgs or HighLevelGoals(goals=[branch.high_level_goal])
    llgs = [
        {"id": f"llg_{index:03d}", **goal.model_dump(mode="json")}
        for index, goal in enumerate(branch.low_level_goals, start=1)
    ]
    sibling_hlgs = [
        goal.model_dump(mode="json")
        for goal in current_hlgs.goals
        if (
            _key(goal.actor.name),
            _key(goal.name),
        ) != (
            _key(branch.high_level_goal.actor.name),
            _key(branch.high_level_goal.name),
        )
    ]
    prompt = f"""Evaluator rationale and calibration:
{reconstruction.rationale}
{LLG_EVALUATOR_EXAMPLES}

Decide conservatively whether the current LLGs contain a material defect in
their decomposition of the valid parent HLG. Regenerate only when at least one
essential capability explicitly required by the parent HLG is absent, or a
current LLG is contradicted by or unsupported in the documentation. Do not
regenerate for naming, style, API/CRUD wording, UI granularity, optional
lifecycle operations, or merely possible improvements. Do not change the HLG.

Each missing capability must be essential to this parent, explicitly grounded
in its wording and documentation, and absent from every current LLG. Do not
import capabilities owned by another HLG. If the evidence is debatable, choose
KEEP_LOW_LEVEL_GOALS. Use REGENERATE only with HIGH confidence.

Complete project description:
{project_description}

Parent HLG:
{branch.high_level_goal.model_dump_json()}

Current LLGs:
{json.dumps(llgs, ensure_ascii=False)}

Sibling HLGs whose responsibilities must not be duplicated:
{json.dumps(sibling_hlgs, ensure_ascii=False)}

Reconstructed HLG:
{reconstruction.reconstructed_high_level_goal}

Use only the supplied llg_NNN identifiers in
unsupported_or_misleading_llg_ids. List concise, explicitly documented parent
capabilities in missing_essential_capabilities. A REGENERATE decision must
contain at least one valid unsupported ID or one missing essential capability.

Output JSON:
{{"rationale":"...","decision":"KEEP_LOW_LEVEL_GOALS | REGENERATE_LOW_LEVEL_GOALS","regeneration_feedback":null,"unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":[],"confidence":"HIGH | MEDIUM | LOW"}}"""
    evaluation = _evaluate(prompt, LowLevelGoalEvaluation)

    valid_ids = {item["id"] for item in llgs}
    supported_ids = [
        item for item in evaluation.unsupported_or_misleading_llg_ids
        if item in valid_ids
    ]
    missing = [
        item.strip()
        for item in evaluation.missing_essential_capabilities
        if item.strip()
    ]
    material_issue = bool(supported_ids or missing)
    if (
        evaluation.decision == LowLevelGoalDecision.REGENERATE
        and (
            evaluation.confidence != ConfidenceLevel.HIGH
            or not material_issue
        )
    ):
        return evaluation.model_copy(update={
            "rationale": (
                f"{evaluation.rationale} Conservative gate: regeneration was "
                "rejected because it lacked HIGH-confidence structured "
                "evidence of a material defect."
            ),
            "decision": LowLevelGoalDecision.KEEP,
            "regeneration_feedback": None,
            "unsupported_or_misleading_llg_ids": supported_ids,
            "missing_essential_capabilities": missing,
        })
    return evaluation.model_copy(update={
        "unsupported_or_misleading_llg_ids": supported_ids,
        "missing_essential_capabilities": missing,
    })


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
Treat each current HLG as a broad stakeholder intention that may cover several
documented workflow steps. Report only the single strongest, clearly distinct
gap with direct documentary support. When overlap is plausible or evidence is
uncertain, return NO_MISSING_HIGH_LEVEL_GOALS. Known actors may be incomplete;
use a new actor only when that actor and intention are explicit in the text.

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
    llg_stabilized: bool = False,
) -> GlobalGoalEvaluationResult:
    try:
        hlg = evaluate_original_hlg(
            project_description, branch, reconstruction, current_hlgs
        )
    except (ValueError, TypeError) as exc:
        return _inconclusive_result(branch, f"HLG evaluation failed: {exc}")
    replacement = None
    llg = None

    if hlg.confidence != ConfidenceLevel.HIGH:
        return _inconclusive_result(
            branch,
            "HLG evaluator confidence was below HIGH; branch was not confirmed.",
            hlg=hlg,
        )

    if hlg.decision == HighLevelGoalDecision.REMOVE:
        decision = GlobalGoalEvaluationDecision.REMOVE_ORIGINAL_HIGH_LEVEL_GOAL
    else:
        if hlg.decision == HighLevelGoalDecision.REWRITE:
            try:
                replacement = build_replacement_request(
                    project_description, branch, reconstruction, hlg
                )
            except (ValueError, TypeError) as exc:
                return _inconclusive_result(
                    branch,
                    f"HLG replacement request failed: {exc}",
                    hlg=hlg,
                )

        if hlg.decision == HighLevelGoalDecision.REWRITE:
            decision = GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
        elif llg_stabilized:
            llg = LowLevelGoalEvaluation(
                rationale=(
                    "The branch reached its bounded LLG repair limit; its latest "
                    "decomposition is retained without another LLG evaluator call."
                ),
                decision=LowLevelGoalDecision.KEEP,
                confidence=ConfidenceLevel.HIGH,
            )
            decision = GlobalGoalEvaluationDecision.LLG_REGENERATION_LIMIT_REACHED
        else:
            try:
                llg = evaluate_llg_decomposition(
                    project_description, branch, reconstruction, current_hlgs
                )
            except (ValueError, TypeError) as exc:
                return _inconclusive_result(
                    branch,
                    f"LLG evaluation failed: {exc}",
                    hlg=hlg,
                )
            if llg.confidence != ConfidenceLevel.HIGH:
                return _inconclusive_result(
                    branch,
                    "LLG evaluator confidence was below HIGH; branch was not confirmed.",
                    hlg=hlg,
                    llg=llg,
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


def _inconclusive_result(
    branch: GoalBranch,
    rationale: str,
    *,
    hlg: HighLevelGoalEvaluation | None = None,
    llg: LowLevelGoalEvaluation | None = None,
) -> GlobalGoalEvaluationResult:
    hlg = hlg or HighLevelGoalEvaluation(
        rationale=rationale,
        decision=HighLevelGoalDecision.KEEP,
        confidence=ConfidenceLevel.LOW,
    )
    return GlobalGoalEvaluationResult(
        branch_id=branch.branch_id,
        rationale=rationale,
        final_decision=GlobalGoalEvaluationDecision.EVALUATION_INCONCLUSIVE,
        high_level_evaluation=hlg,
        low_level_evaluation=llg,
    )


def evaluate_all_branches(
    project_description: str,
    branches: list[GoalBranch],
    reconstructions: dict[str, BottomUpHighLevelGoal],
    current_hlgs: HighLevelGoals,
    stabilized_hlg_keys: set[str] | None = None,
):
    stabilized_hlg_keys = stabilized_hlg_keys or set()
    return {
        branch.branch_id: evaluate_branch(
            project_description,
            branch,
            reconstructions[branch.branch_id],
            current_hlgs,
            _stable_hlg_key(branch) in stabilized_hlg_keys,
        )
        for branch in branches
    }
