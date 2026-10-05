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
    MissingHighLevelGoalDecision,
    MissingHighLevelGoalEvaluation,
)
from src.llm_clients import EvaluatorConversation, generate_evaluator_response


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

HLG_EVALUATOR_EXAMPLES = """Few-shot examples:
- Description: customers browse products and place orders. Original HLG:
  "Purchase products"; reconstruction: "Find and order products" ->
  KEEP_ORIGINAL_HIGH_LEVEL_GOAL,
  because both express the same supported customer intention.
- Description: organizers create and manage events. Original HLG: "Click the
  create-event button"; reconstruction: "Plan and manage events" ->
  REWRITE_ORIGINAL_HIGH_LEVEL_GOAL,
  because the original is a UI step rather than the actor's WHY.
- Description contains no refunds and another same-actor HLG already covers
  order management. Original HLG: "Manage refunds" ->
  REMOVE_ORIGINAL_HIGH_LEVEL_GOAL.
"""
LLG_EVALUATOR_EXAMPLES = """Few-shot examples:
- Parent: "Manage profile". LLGs update contact data and upload a profile image;
  both are documented and API-mappable -> KEEP_LOW_LEVEL_GOALS.
- Parent: "Submit and track an application". LLGs submit it but provide no way
  to retrieve its documented status -> REGENERATE_LOW_LEVEL_GOALS and list
  status retrieval as a missing essential capability.
- Parent: "Browse products". One LLG cancels paid orders, while cancellation is
  undocumented and belongs to order management -> REGENERATE_LOW_LEVEL_GOALS
  and identify it.
- LLG wording says "call endpoint" but covers a required action ->
  KEEP_LOW_LEVEL_GOALS; wording or optional CRUD improvements alone are not
  material defects.
"""
COVERAGE_EVALUATOR_EXAMPLES = """Few-shot examples:
- Description says a Customer searches a catalogue, compares products, and
  places orders. Current HLG "Discover and purchase products" covers the whole
  intention -> NO_MISSING_HIGH_LEVEL_GOALS; do not add narrower search/compare.
- Description repeatedly assigns an Operator the review, approval, and rejection
  of applications, but no Operator HLG exists -> MISSING_HIGH_LEVEL_GOALS_FOUND
  with request "Evaluate applications", even if that exact title is not stated.
- Description requires an existing Applicant to upload requested documents after
  suspension, while current HLGs cover only submission and status monitoring ->
  MISSING_HIGH_LEVEL_GOALS_FOUND because this is a distinct responsibility.
- Audit logging is common in similar systems but absent from the description ->
  NO_MISSING_HIGH_LEVEL_GOALS. Never invent a new actor.
"""


def _json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Evaluator response does not contain a JSON object.")
    return json.loads(cleaned[start : end + 1])


def _evaluate(
    prompt: str,
    model: type[BaseModel],
    conversation: EvaluatorConversation | None = None,
    memory_label: str | None = None,
):
    response = generate_evaluator_response(
        prompt,
        SYSTEM_PROMPT,
        conversation=conversation,
    )
    try:
        result = model.model_validate(_json_object(response))
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
    if conversation is not None:
        conversation.record_exchange(
            memory_label or f"{model.__name__} evaluation",
            response,
        )
    return result


def evaluate_original_hlg(
    project_description: str,
    branch: GoalBranch,
    reconstruction: BottomUpHighLevelGoal,
    current_hlgs: HighLevelGoals | None = None,
    conversation: EvaluatorConversation | None = None,
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
    return _evaluate(
        prompt,
        HighLevelGoalEvaluation,
        conversation,
        memory_label=(
            "HLG evaluation for actor "
            f"'{branch.high_level_goal.actor.name}', goal "
            f"'{branch.high_level_goal.name}'"
        ),
    )


def build_replacement_request(
    project_description: str,
    branch: GoalBranch,
    evaluation: HighLevelGoalEvaluation,
) -> HighLevelGoalReplacementRequest:
    """Build the generator request from the HLG evaluator output."""
    return HighLevelGoalReplacementRequest(
        rationale=evaluation.rationale,
        generation_project_description=(
            f"{project_description}\n\n"
            f"Rewrite the functional intention for actor "
            f"'{branch.high_level_goal.actor.name}' using this focus: "
            f"{evaluation.rewriting_focus}"
        ),
        actor=branch.high_level_goal.actor,
    )


def evaluate_llg_decomposition(
    project_description: str,
    branch: GoalBranch,
    reconstruction: BottomUpHighLevelGoal,
    current_hlgs: HighLevelGoals | None = None,
    conversation: EvaluatorConversation | None = None,
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
    evaluation = _evaluate(
        prompt,
        LowLevelGoalEvaluation,
        conversation,
        memory_label=(
            "LLG decomposition evaluation for parent "
            f"'{branch.high_level_goal.name}'"
        ),
    )

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
    conversation: EvaluatorConversation | None = None,
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
documented workflow steps. A missing intention may be explicit or may emerge
from multiple passages, the end-to-end workflow, and an existing actor's
responsibilities. Such an inference must remain grounded in the description:
do not add a goal merely because it is conventional or common in similar
  systems. Report up to two strong, clearly distinct gaps. A goal does not need
  to appear as one explicit sentence: report it when at least two coherent
  workflow passages or actor responsibilities support the same functional WHY.
  Return NO_MISSING_HIGH_LEVEL_GOALS when a current broad HLG plausibly covers
  the intention or the evidence is only conventional domain knowledge.
The actor must be selected from the supplied existing actor list; never
introduce a new actor.

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
    evaluation = _evaluate(
        prompt,
        MissingHighLevelGoalEvaluation,
        conversation,
        memory_label=(
            "Project-wide HLG coverage evaluation with "
            f"{len(current_hlgs.goals)} current HLGs"
        ),
    )
    return _restrict_missing_goals_to_existing_actors(evaluation, actors)


def _restrict_missing_goals_to_existing_actors(
    evaluation: MissingHighLevelGoalEvaluation,
    actors: Actors,
) -> MissingHighLevelGoalEvaluation:
    """Keep up to two distinct requests whose actors already exist."""
    actors_by_name = {_key(actor.name): actor.name for actor in actors.actors}
    valid_requests = []
    for request in evaluation.missing_goal_requests:
        actor_name = actors_by_name.get(_key(request.actor))
        if actor_name is None:
            continue
        normalized_request = request.model_copy(update={"actor": actor_name})
        request_key = (
            _key(normalized_request.actor),
            _key(normalized_request.generation_project_description),
        )
        if any(
            (_key(item.actor), _key(item.generation_project_description))
            == request_key
            for item in valid_requests
        ):
            continue
        valid_requests.append(normalized_request)
        if len(valid_requests) >= 2:
            break

    if not valid_requests:
        return MissingHighLevelGoalEvaluation(
            decision=MissingHighLevelGoalDecision.NO_MISSING,
            rationale=(
                evaluation.rationale
                if not evaluation.missing_goal_requests
                else (
                    f"{evaluation.rationale} Proposed requests were discarded "
                    "because they did not reference an existing actor."
                )
            ),
            missing_goal_requests=[],
        )
    return evaluation.model_copy(update={"missing_goal_requests": valid_requests})


def evaluate_branch(
    project_description: str,
    branch: GoalBranch,
    reconstruction: BottomUpHighLevelGoal,
    current_hlgs: HighLevelGoals,
    llg_stabilized: bool = False,
    conversation: EvaluatorConversation | None = None,
) -> GlobalGoalEvaluationResult:
    try:
        hlg = evaluate_original_hlg(
            project_description,
            branch,
            reconstruction,
            current_hlgs,
            conversation,
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
                    project_description, branch, hlg
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
                    project_description,
                    branch,
                    reconstruction,
                    current_hlgs,
                    conversation,
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
    conversation: EvaluatorConversation | None = None,
):
    stabilized_hlg_keys = stabilized_hlg_keys or set()
    return {
        branch.branch_id: evaluate_branch(
            project_description,
            branch,
            reconstructions[branch.branch_id],
            current_hlgs,
            _stable_hlg_key(branch) in stabilized_hlg_keys,
            conversation,
        )
        for branch in branches
    }
