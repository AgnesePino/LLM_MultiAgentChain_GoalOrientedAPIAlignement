"""Sequential branch and global bottom-up evaluations with deterministic aggregation."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re

from pydantic import BaseModel

from src.data_model import (
    BottomUpHighLevelGoal,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationResult,
    GoalBranch,
    HighLevelGoalDecision,
    HighLevelGoalEvaluation,
    HighLevelGoalRemovalBasis,
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

BOTTOM_UP_EVALUATOR_VOTERS = int(
    os.getenv("BOTTOM_UP_EVALUATOR_VOTERS", "3")
)
BOTTOM_UP_EVALUATOR_TEMPERATURE = float(
    os.getenv("BOTTOM_UP_EVALUATOR_TEMPERATURE", "0.2")
)
BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS = int(
    os.getenv("BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS", "3")
)
if BOTTOM_UP_EVALUATOR_VOTERS not in {3, 5}:
    raise ValueError("BOTTOM_UP_EVALUATOR_VOTERS must be either 3 or 5.")
if not 1 <= BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS <= 3:
    raise ValueError(
        "BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS must be between 1 and 3."
    )
EVALUATOR_VOTING_QUORUM = BOTTOM_UP_EVALUATOR_VOTERS // 2 + 1
BOTTOM_UP_QUALITY_THRESHOLD = 3


def _key(value: str) -> str:
    return " ".join(value.casefold().split())


def _other_high_level_goals(
    current_hlgs: HighLevelGoals,
    branch: GoalBranch,
) -> list[dict]:
    """Exclude only the current HLG while preserving duplicates for review."""
    skipped_current = False
    siblings = []
    for goal in current_hlgs.goals:
        if not skipped_current and goal == branch.high_level_goal:
            skipped_current = True
            continue
        siblings.append(goal.model_dump(mode="json"))
    return siblings


HLG_EVALUATOR_EXAMPLES = """Few-shot examples:
- Correct goal. Description: CatWatch shows repository popularity and active
  contributors. Original HLG: "Monitor GitHub project activity"; reconstruction:
  "Analyze repository popularity and contributors".
  Output: {"rationale":"Both goals express the same documented stakeholder intention.","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":5,"rewriting_focus":null,"removal_basis":null,"covered_by_high_level_goal_name":null}
- Modification error. Description: CatWatch fetches GitHub statistics
  automatically. Original HLG: "Manually enter repository statistics";
  reconstruction: "Collect and analyze GitHub statistics".
  Output: {"rationale":"Manual entry contradicts the documented automatic collection, but the underlying analytics intention is valid.","decision":"REWRITE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":2,"rewriting_focus":"Express automatic collection and analysis of GitHub statistics.","removal_basis":null,"covered_by_high_level_goal_name":null}
- Erroneous addition. Description: CatWatch only collects and reports GitHub
  statistics. Original HLG: "Pay contributors"; reconstruction: "Process
  contributor payments".
  Output: {"rationale":"The payment intention is unsupported by the description in both the original and reconstructed goals.","decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":0,"rewriting_focus":null,"removal_basis":"UNSUPPORTED","covered_by_high_level_goal_name":null}
- Redundant workflow fragment. Description: customers create, submit, and track
  orders. Original HLG: "Track submitted orders"; another HLG for Customer is
  "Manage the order lifecycle", covering creation through tracking.
  Output: {"rationale":"Tracking is a documented workflow phase already encompassed by the broader sibling intention.","decision":"REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":1,"rewriting_focus":null,"removal_basis":"FULLY_REDUNDANT","covered_by_high_level_goal_name":"Manage the order lifecycle"}
- Fragmented workflow without an umbrella. Description: editors draft, revise,
  publish, and archive one article. Original HLG: "Publish an article";
  siblings separately cover drafting and archiving, but no HLG expresses the
  complete editor intention.
  Output: {"rationale":"The goal is supported but too narrow: it is one phase of a single end-to-end content-management intention.","decision":"REWRITE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":2,"rewriting_focus":"Express one end-to-end goal covering drafting, revision, publication, and archival of an article.","removal_basis":null,"covered_by_high_level_goal_name":null}
"""
LLG_EVALUATOR_EXAMPLES = """Few-shot examples:
- Correct decomposition. Parent: "Monitor repository popularity". LLGs retrieve
  stars and forks and display their trends.
  Output: {"rationale":"The LLGs cover the documented parent capability.","decision":"KEEP_LOW_LEVEL_GOALS","quality_score":5,"regeneration_feedback":null,"unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":[]}
- Major omission. Parent: "Monitor popularity and active contributors". LLGs:
  llg_001 retrieves stars and llg_002 displays popularity trends.
  Output: {"rationale":"Contributor monitoring is explicit in the parent but absent from every LLG.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":1,"regeneration_feedback":"Keep the popularity LLGs and add contributor monitoring.","unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":["Identify active repository contributors"]}
- Erroneous addition. Parent: "Analyze repository activity". llg_001 retrieves
  commits; llg_002 publishes a marketing post, which is undocumented.
  Output: {"rationale":"The marketing action is unsupported and outside the parent goal.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":1,"regeneration_feedback":"Keep repository analysis and remove the marketing action.","unsupported_or_misleading_llg_ids":["llg_002"],"missing_essential_capabilities":[]}
- Conventional but undocumented addition. Description: users submit support
  requests. Parent: "Obtain support". llg_001 submits a request and llg_002
  configures automatic status notifications; notifications are not documented.
  Output: {"rationale":"Automatic notifications are plausible but not entailed by the supplied documentation.","decision":"REGENERATE_LOW_LEVEL_GOALS","quality_score":2,"regeneration_feedback":"Keep request submission and remove the unsupported notification capability.","unsupported_or_misleading_llg_ids":["llg_002"],"missing_essential_capabilities":[]}
"""
COVERAGE_EVALUATOR_EXAMPLES = """Few-shot examples:
- Correct coverage. Description: customers browse menus and place orders.
  Current HLG: "Find and order food" for Customer.
  Output: {"decision":"NO_MISSING_HIGH_LEVEL_GOALS","rationale":"Browsing and ordering are steps of the existing end-to-end intention.","quality_score":5,"missing_goal_requests":[]}
- Major omission. Description: customers place orders and couriers update
  delivery status. Current HLGs cover only Customer; Courier is an existing actor.
  Output: {"decision":"MISSING_HIGH_LEVEL_GOALS_FOUND","rationale":"The Courier has an explicit responsibility not covered by any current HLG.","quality_score":1,"missing_goal_requests":[{"actor":"Courier","generation_project_description":"Generate one HLG for managing assigned deliveries and their status."}]}
- Erroneous addition. Description: a librarian registers loans and returns.
  Current HLG: "Manage book circulation". Audit logging is not documented.
  Output: {"decision":"NO_MISSING_HIGH_LEVEL_GOALS","rationale":"The current HLG covers loans and returns; conventional audit features are not evidence of a missing goal.","quality_score":4,"missing_goal_requests":[]}
"""

QUALITY_SCORE_RUBRIC = """Quality-score rubric for the current artifact:
- 0: unsupported or fundamentally contradictory;
- 1: severe omission or unsupported content requiring correction;
- 2: material defect, but the main intention remains recoverable;
- 3: acceptable and supported, with only minor non-material defects;
- 4: good coverage and alignment, with negligible issues;
- 5: complete, precise, supported, and internally consistent.
Scores 3, 4, and 5 pass. Scores 0, 1, and 2 require a corrective decision.
The score evaluates the artifact, not confidence in your own answer."""


def _json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Evaluator response does not contain a JSON object.")
    return json.loads(cleaned[start : end + 1])


def _score_matches_decision(evaluation: BaseModel) -> bool:
    """Accept only decisions that apply the bottom-up quality threshold."""
    passing = evaluation.quality_score >= BOTTOM_UP_QUALITY_THRESHOLD
    if evaluation.decision in {
        HighLevelGoalDecision.KEEP,
        LowLevelGoalDecision.KEEP,
        MissingHighLevelGoalDecision.NO_MISSING,
    }:
        return passing
    if evaluation.decision in {
        HighLevelGoalDecision.REWRITE,
        HighLevelGoalDecision.REMOVE,
        LowLevelGoalDecision.REGENERATE,
        MissingHighLevelGoalDecision.FOUND,
    }:
        return not passing
    return False


def _evaluate(
    prompt: str,
    model: type[BaseModel],
    conversation: EvaluatorConversation | None = None,
    memory_label: str | None = None,
):
    def request_vote():
        last_error = "unknown evaluator error"
        for _ in range(BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS):
            try:
                response = generate_evaluator_response(
                    prompt,
                    SYSTEM_PROMPT,
                    conversation=conversation,
                    temperature=BOTTOM_UP_EVALUATOR_TEMPERATURE,
                )
                evaluation = model.model_validate(_json_object(response))
                if not _score_matches_decision(evaluation):
                    raise ValueError(
                        "quality_score conflicts with the selected decision"
                    )
                return evaluation, None
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        return None, last_error

    with ThreadPoolExecutor(max_workers=BOTTOM_UP_EVALUATOR_VOTERS) as executor:
        vote_results = list(
            executor.map(
                lambda _: request_vote(),
                range(BOTTOM_UP_EVALUATOR_VOTERS),
            )
        )

    valid_votes = [vote for vote, _ in vote_results if vote is not None]
    rejected_reasons = Counter(
        error for vote, error in vote_results if vote is None and error
    )
    rejection_summary = "; ".join(
        f"{count}x {reason}" for reason, count in rejected_reasons.items()
    )
    decision_counts = Counter(vote.decision for vote in valid_votes)
    if not decision_counts:
        raise ValueError(
            f"No valid {model.__name__} votes were returned by the evaluator. "
            f"Rejected votes: {rejection_summary or 'none recorded'}."
        )
    winning_decision, winning_count = decision_counts.most_common(1)[0]
    if winning_count < EVALUATOR_VOTING_QUORUM:
        readable_counts = {
            decision.value: count for decision, count in decision_counts.items()
        }
        raise ValueError(
            f"No {model.__name__} voting quorum: {readable_counts}; "
            f"required {EVALUATOR_VOTING_QUORUM}/{BOTTOM_UP_EVALUATOR_VOTERS}. "
            f"Rejected votes: {rejection_summary or 'none'}."
        )

    winning_votes = [
        vote for vote in valid_votes if vote.decision == winning_decision
    ]
    representative = max(
        winning_votes,
        key=lambda vote: (
            len(getattr(vote, "unsupported_or_misleading_llg_ids", []))
            + len(getattr(vote, "missing_essential_capabilities", []))
            + len(getattr(vote, "missing_goal_requests", []))
        ),
    )
    result = representative.model_copy(update={
        "rationale": (
            f"Voting result {winning_count}/{BOTTOM_UP_EVALUATOR_VOTERS} "
            f"for {winning_decision.value}. {representative.rationale}"
        ),
        "voter_count": BOTTOM_UP_EVALUATOR_VOTERS,
        "valid_vote_count": len(valid_votes),
        "winning_vote_count": winning_count,
        "vote_distribution": {
            decision.value: count for decision, count in decision_counts.items()
        },
    })
    if conversation is not None:
        conversation.record_exchange(
            memory_label or f"{model.__name__} evaluation",
            result.model_dump_json(),
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
    sibling_hlgs = _other_high_level_goals(current_hlgs, branch)
    prompt = f"""Evaluator rationale from the bottom-up reconstruction:
{reconstruction.rationale}

Evaluator calibration:
{HLG_EVALUATOR_EXAMPLES}

{QUALITY_SCORE_RUBRIC}

Decide whether the original High-Level Goal should be kept, rewritten, or
removed. The project description is the source of truth. Documentation support
is necessary but not sufficient: the HLG must also use the same intentional
granularity as the stakeholder outcome. A workflow phase, state transition,
validation step, exception path, or CRUD capability is not automatically an
independent HLG merely because it is explicitly documented.

If the branch has no LLGs, treat that as a decomposition omission, not as
evidence that the original HLG is unsupported. Judge KEEP, REWRITE, or REMOVE
from the project description and sibling hierarchy exactly as for a non-empty
branch. A valid empty HLG must be kept or rewritten so that its LLGs can be
generated by the subsequent repair step.

Perform this hierarchy-wide consolidation audit before deciding:
1. Identify the end-to-end functional outcome pursued by this actor.
2. Group documented capabilities that operate on the same artifact and jointly
   realize that outcome. Do not split creation, draft handling, submission,
   consultation, validation, and exception handling into separate HLGs unless
   the description presents independently satisfiable stakeholder outcomes.
3. Choose REMOVE only when the goal is wholly unsupported, or when one named
   sibling for the same actor already entails every documented capability and
   every distinctive LLG of the original goal. Partial overlap, a shared domain
   object, or a broader-sounding label is not sufficient.
4. If the hierarchy is fragmented and no sibling is yet a valid umbrella,
   choose REWRITE for the broadest suitable goal so it can become that umbrella.
5. Choose KEEP only when the goal is both supported and independently scoped,
   not simply a distinct step of a sibling's workflow.

Rewrite an HLG when it is too generic, narrow, ambiguous, incorrectly scoped,
or less faithful than the reconstruction. Remove it when documentation does not
support it or another HLG for the same actor demonstrably covers its complete
intention. When a supported goal contains any distinct stakeholder outcome or
capability not preserved by one sibling, prefer KEEP or REWRITE over REMOVE.
Judge semantic entailment, not equal names or superficial thematic overlap. Do
not justify KEEP merely by saying that two goals concern different workflow
stages. A supported but materially over-fragmented goal must receive score 0-2
and a corrective decision, not KEEP.

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

For REMOVE, set removal_basis to UNSUPPORTED or FULLY_REDUNDANT. A
FULLY_REDUNDANT removal must name exactly one surviving sibling in
covered_by_high_level_goal_name. Otherwise use null for both fields.

Output JSON:
{{"rationale":"...","decision":"KEEP_ORIGINAL_HIGH_LEVEL_GOAL | REWRITE_ORIGINAL_HIGH_LEVEL_GOAL | REMOVE_ORIGINAL_HIGH_LEVEL_GOAL","quality_score":0,"rewriting_focus":null,"removal_basis":null,"covered_by_high_level_goal_name":null}}"""
    evaluation = _evaluate(
        prompt,
        HighLevelGoalEvaluation,
        conversation,
        memory_label=(
            "HLG evaluation for actor "
            f"'{branch.high_level_goal.actor.name}', goal "
            f"'{branch.high_level_goal.name}'"
        ),
    )
    removal_is_supported = (
        evaluation.removal_basis == HighLevelGoalRemovalBasis.UNSUPPORTED
        or (
            evaluation.removal_basis
            == HighLevelGoalRemovalBasis.FULLY_REDUNDANT
            and any(
                _key(sibling.get("name", ""))
                == _key(evaluation.covered_by_high_level_goal_name or "")
                and _key((sibling.get("actor") or {}).get("name", ""))
                == _key(branch.high_level_goal.actor.name)
                for sibling in sibling_hlgs
            )
        )
    )
    if evaluation.decision == HighLevelGoalDecision.REMOVE and (
        evaluation.winning_vote_count < evaluation.voter_count
        or not removal_is_supported
    ):
        reasons = []
        if evaluation.winning_vote_count < evaluation.voter_count:
            reasons.append("REMOVE was not unanimous")
        if not removal_is_supported:
            reasons.append(
                "no valid unsupported/redundant removal basis with a named "
                "same-actor sibling was supplied"
            )
        return evaluation.model_copy(update={
            "rationale": (
                f"{evaluation.rationale} Destructive-action gate: "
                f"{'; '.join(reasons)}; the HLG is retained in this state."
            ),
            "decision": HighLevelGoalDecision.KEEP,
            "quality_score": BOTTOM_UP_QUALITY_THRESHOLD,
            "rewriting_focus": None,
            "removal_basis": None,
            "covered_by_high_level_goal_name": None,
        })
    return evaluation


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
    sibling_hlgs = _other_high_level_goals(current_hlgs, branch)
    prompt = f"""Evaluator rationale and calibration:
{reconstruction.rationale}
{LLG_EVALUATOR_EXAMPLES}

{QUALITY_SCORE_RUBRIC}

Decide conservatively whether the current LLGs contain a material defect in
their decomposition of the valid parent HLG. Regenerate when at least one
essential capability explicitly required by the parent HLG is absent, or a
current LLG is contradicted by or unsupported in the documentation. Do not
regenerate merely because an optional operation is absent, but do regenerate
when an optional operation is already present and lacks documentary support.
Do not regenerate only for naming, style, API/CRUD wording, UI granularity, or
merely possible improvements. Do not change the HLG.

Audit every current LLG individually. Its capability must be explicitly stated
or necessarily entailed by the project description and the parent HLG. Being
common, useful, conventional, standard for the domain, or compatible with the
workflow is not evidence. Do not infer notifications, downloads, reassignment,
internal notes, audit operations, confirmations, status transitions, or CRUD
variants unless the supplied description entails them. Broad verbs such as
"manage", "monitor", or "process" do not authorize arbitrary lifecycle
operations. Put every unsupported current LLG in
unsupported_or_misleading_llg_ids and choose REGENERATE.

Each missing capability must be essential to this parent, explicitly grounded
in its wording and documentation, and absent from every current LLG. Do not
import capabilities owned by another HLG. If the evidence is debatable, choose
REGENERATE_LOW_LEVEL_GOALS and identify the unsupported current LLG or the
explicitly missing essential capability. Uncertainty must not preserve invented
functionality.

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
{{"rationale":"...","decision":"KEEP_LOW_LEVEL_GOALS | REGENERATE_LOW_LEVEL_GOALS","quality_score":0,"regeneration_feedback":null,"unsupported_or_misleading_llg_ids":[],"missing_essential_capabilities":[]}}"""
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
        and not material_issue
    ):
        return evaluation.model_copy(update={
            "rationale": (
                f"{evaluation.rationale} Conservative gate: regeneration was "
                "rejected because it lacked structured evidence of a "
                "material defect."
            ),
            "decision": LowLevelGoalDecision.KEEP,
            "quality_score": BOTTOM_UP_QUALITY_THRESHOLD,
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
    retired_hlgs: HighLevelGoals | None = None,
) -> MissingHighLevelGoalEvaluation:
    """Check project-wide HLG coverage once per bottom-up iteration."""
    retired_hlgs = retired_hlgs or HighLevelGoals(goals=[])
    prompt = f"""Evaluator rationale and calibration:
{COVERAGE_EVALUATOR_EXAMPLES}

{QUALITY_SCORE_RUBRIC}

Perform a fresh requirement-by-requirement coverage audit. Check whether the
project description contains functional, actor-level High-Level Goals that are
missing from the current HLG list. Previous conversation conclusions are
context only, not evidence; reassess coverage from the supplied description and
current HLGs on every call.

Do not generate final HLG objects. Return only focused requests for the normal
top-down HLG generator. Do not propose goals already covered by current HLGs,
technical/API-level operations, or intentions unsupported by the description.

Use this audit procedure internally before deciding:
1. Extract every explicit actor capability, desired outcome, mandatory
   interaction, and externally visible responsibility from the description.
2. Assign each requirement to one supplied actor and one current HLG only when
   that HLG's wording semantically entails the same outcome.
3. List requirements left uncovered, then group only those that serve one
   coherent stakeholder WHY.
4. Report every distinct documented actor-level gap. Do not truncate the list
   to an arbitrary number and do not merge independent outcomes merely to keep
   the response short.

A current HLG may cover several workflow steps, CRUD operations, or narrower
variants that jointly realize the same outcome. However, thematic similarity is
not coverage. Do not treat a requirement as covered merely because it concerns
the same domain object, actor, workflow, or broad topic. Separate it when it has
an independently satisfiable outcome, a distinct interaction channel or
external-system integration, a different artifact being managed, or a separate
actor responsibility that would require its own coherent LLG decomposition.

One clear, explicit functional requirement is sufficient evidence for a gap; it
does not need to be repeated in two passages. An implicit gap still requires at
least two coherent workflow passages or responsibilities. Never infer a feature
from conventional domain knowledge. Return NO_MISSING_HIGH_LEVEL_GOALS only
when every explicit functional requirement is entailed by a current HLG or the
remaining evidence is non-functional, implementation-only, ambiguous, or
merely conventional.

For every proposed gap, use the actor name exactly as it appears in the supplied
actor list. The generation_project_description must identify the uncovered
outcome and cite its concrete support from the project description; it must not
request unrelated improvements or multiple intentions. Never introduce a new
actor.

Complete project description:
{project_description}

Already identified actors:
{actors.model_dump_json()}

Current High-Level Goals:
{json.dumps([goal.model_dump(mode="json") for goal in current_hlgs.goals], ensure_ascii=False)}

HLGs removed or replaced earlier in this cycle:
{json.dumps([goal.model_dump(mode="json") for goal in retired_hlgs.goals], ensure_ascii=False)}

Audit every responsibility represented by the retired HLGs. Do not restore a
retired goal when a current HLG already entails its documented outcome, but do
report a gap when deletion or rewriting left that outcome uncovered. The
retired list is an audit checklist, not evidence that a goal must be restored.

Output only valid JSON:
{{
  "decision": "NO_MISSING_HIGH_LEVEL_GOALS | MISSING_HIGH_LEVEL_GOALS_FOUND",
  "rationale": "...",
  "quality_score": 0,
  "missing_goal_requests": [
    {{"actor": "Actor name", "generation_project_description": "Focused stakeholder description"}}
  ]
}}"""
    try:
        evaluation = _evaluate(
            prompt,
            MissingHighLevelGoalEvaluation,
            conversation,
            memory_label=(
                "Project-wide HLG coverage evaluation with "
                f"{len(current_hlgs.goals)} current HLGs"
            ),
        )
    except (ValueError, TypeError) as exc:
        return MissingHighLevelGoalEvaluation(
            decision=MissingHighLevelGoalDecision.INCONCLUSIVE,
            rationale=f"Coverage voting failed: {exc}",
            quality_score=0,
            missing_goal_requests=[],
            voter_count=BOTTOM_UP_EVALUATOR_VOTERS,
            valid_vote_count=0,
            winning_vote_count=0,
            vote_distribution={},
        )
    return _restrict_missing_goals_to_existing_actors(evaluation, actors)


def _restrict_missing_goals_to_existing_actors(
    evaluation: MissingHighLevelGoalEvaluation,
    actors: Actors,
) -> MissingHighLevelGoalEvaluation:
    """Keep all distinct requests whose actors already exist."""
    if evaluation.decision == MissingHighLevelGoalDecision.INCONCLUSIVE:
        return evaluation
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

    if not valid_requests:
        if evaluation.decision == MissingHighLevelGoalDecision.NO_MISSING:
            return evaluation
        return evaluation.model_copy(update={
            "decision": MissingHighLevelGoalDecision.INCONCLUSIVE,
            "rationale": (
                f"{evaluation.rationale} Proposed requests were discarded "
                "because they did not reference an existing actor."
            ),
            "quality_score": 0,
            "missing_goal_requests": [],
        })
    return evaluation.model_copy(update={"missing_goal_requests": valid_requests})


def evaluate_branch(
    project_description: str,
    branch: GoalBranch,
    reconstruction: BottomUpHighLevelGoal,
    current_hlgs: HighLevelGoals,
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
            decision = (
                GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS
                if llg.decision == LowLevelGoalDecision.REGENERATE
                else GlobalGoalEvaluationDecision.CONFIRM_BRANCH
            )
            if not branch.low_level_goals and decision == GlobalGoalEvaluationDecision.CONFIRM_BRANCH:
                llg = llg.model_copy(update={
                    "rationale": (
                        f"{llg.rationale} Empty-branch invariant: a valid HLG "
                        "cannot be confirmed without an LLG decomposition."
                    ),
                    "decision": LowLevelGoalDecision.REGENERATE,
                    "quality_score": min(llg.quality_score, 2),
                    "regeneration_feedback": (
                        "Generate a complete, non-redundant decomposition of "
                        "the documented parent HLG into atomic functional "
                        "interactions."
                    ),
                    "missing_essential_capabilities": [
                        branch.high_level_goal.description
                    ],
                })
                decision = GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS

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
        quality_score=0,
        voter_count=BOTTOM_UP_EVALUATOR_VOTERS,
        valid_vote_count=0,
        winning_vote_count=0,
        vote_distribution={},
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
    conversation: EvaluatorConversation | None = None,
):
    return {
        branch.branch_id: evaluate_branch(
            project_description,
            branch,
            reconstructions[branch.branch_id],
            current_hlgs,
            conversation,
        )
        for branch in branches
    }
