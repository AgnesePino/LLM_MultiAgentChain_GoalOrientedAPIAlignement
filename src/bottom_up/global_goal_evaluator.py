"""
Global Goal Evaluator for the bottom-up feedback loop.

For each non-empty branch, a separate bottom-up reconstructor has already
inferred one High-Level Goal (HLG') from that branch's Low-Level Goals (LLGs).
The branch evaluator intentionally receives only:
- the complete project documentation;
- the complete current HLG collection;
- the current branch id, which identifies the parent HLG in that collection;
- the HLG' reconstructed bottom-up for that branch.

The evaluator does not inspect individual LLGs and does not distinguish an
"HLG problem" from an "LLG problem". It performs one round-trip verification:

    current HLG -> current LLGs -> reconstructed HLG'

Branch evaluation is decomposed into up to three focused LLM stages but has exactly two final outcomes:
- CONFIRM_BRANCH: HLG' sufficiently preserves the documented intention of its
  current parent;
- REGENERATE_HIGH_LEVEL_GOAL: the branch is not stable. The current parent HLG
  is regenerated through the ORIGINAL HLG Generator -> HLG Evaluator loop and
  the replacement HLG ALWAYS receives a new LLG decomposition through the
  ORIGINAL LLG Generator -> LLG Evaluator loop before the next iteration.

Only after every current branch is confirmed does the separate documentation-
coverage stage run. Coverage uses the documentation and the current HLG set as
its source of truth and may also inspect the confirmed reconstructed HLGs as
signals of documented intentions that emerged from the LLGs but are not yet
represented by an autonomous current HLG. Such a signal never creates a goal
by itself: a new HLG is added only when the documentation supports the
intention, no current HLG already covers it, and it is genuinely WHY-level.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

from src.data_model import (
    Actors,
    BottomUpHighLevelGoal,
    DocumentationCoverageLLMOutput,
    DocumentationCoverageResult,
    DocumentationCoverageSelectionLLMOutput,
    DocumentationCoverageStatus,
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationLLMOutput,
    GlobalGoalEvaluationResult,
    HighLevelGoal,
    HighLevelGoalGenerationAction,
    HighLevelGoalGenerationRequest,
    HighLevelGoalGeneratorInput,
    HighLevelGoalGenerationSource,
    HighLevelGoals,
)
from src.bottom_up.goal_reconstructor import normalize_goal_name
from src.bottom_up.semantic_similarity import find_semantic_duplicate_high_level_goal
from src.llm_clients import generate_response_llama, MAX_SEMANTIC_RETRIES


class GlobalGoalEvaluationError(ValueError):
    """Raised when a branch-evaluation response is invalid or inconsistent."""


_OPERATIONAL_ACTION_BY_DECISION: dict[GlobalGoalEvaluationDecision, str] = {
    GlobalGoalEvaluationDecision.CONFIRM_BRANCH: "KEEP_BRANCH",
    GlobalGoalEvaluationDecision.REGENERATE_HIGH_LEVEL_GOAL: (
        "REGENERATE_HLG_AND_THEN_REGENERATE_ITS_LLGS"
    ),
}

_BRANCH_ALLOWED_DECISIONS = {
    GlobalGoalEvaluationDecision.CONFIRM_BRANCH,
    GlobalGoalEvaluationDecision.REGENERATE_HIGH_LEVEL_GOAL,
}

_GORE_EXPERT_PREAMBLE = """You are a Requirements Engineering evaluator specialized in Goal-Oriented Requirements Engineering (GORE). Evaluate functional stakeholder intentions at WHY level and ground every judgment in the provided project documentation.

"""

_GENERATION_DESCRIPTION_RULES = """# Regeneration input
When regeneration is required, generation_project_description must be 1-3 concise stakeholder-style sentences describing exactly one documented functional WHY-level intention for the current branch. Treat the project documentation as the factual source; use the parent HLG and HLG' only to identify the branch scope. Omit branch ids, pipeline language, critique, implementation details, quality attributes, and any final HLG name.

"""

def _format_existing_high_level_goals_block(
    existing_high_level_goals: dict[str, HighLevelGoal],
) -> str:
    """Render the complete current HLG collection with opaque branch ids."""
    if not existing_high_level_goals:
        return "(none)"

    return "\n".join(
        f"- branch_id: {branch_id}\n"
        f"  goal_name: {goal.name}\n"
        f"  description: {goal.description}\n"
        f"  actor: {goal.actor.name}"
        for branch_id, goal in existing_high_level_goals.items()
    )


def _format_reconstructed_high_level_goals_block(
    reconstructed_high_level_goals: dict[str, BottomUpHighLevelGoal] | None,
) -> str:
    """Render confirmed bottom-up HLG reconstructions as coverage signals."""
    if not reconstructed_high_level_goals:
        return "(none)"

    return "\n".join(
        f"- branch_id: {branch_id}\n"
        f"  reconstructed_hlg: {goal.reconstructed_high_level_goal}"
        for branch_id, goal in reconstructed_high_level_goals.items()
    )


class BranchGroundingLLMOutput(BaseModel):
    """Stage 1: documentation grounding for the parent HLG and HLG'."""

    rationale: str
    parent_supported: bool
    reconstructed_supported: bool
    documented_branch_intention: str | None = None


class BranchAlignmentLLMOutput(BaseModel):
    """Stage 2: semantic round-trip alignment between parent HLG and HLG'."""

    rationale: str
    preserves_parent_intention: bool


def _build_branch_grounding_system_prompt() -> str:
    """Stage 1: ground parent and reconstructed HLG in the documentation."""
    return _GORE_EXPERT_PREAMBLE + """# Stage 1 - Documentation grounding
Assess only whether the current parent HLG and the reconstructed HLG' are supported by the project documentation.

The parent is the HLG whose branch_id equals current_branch_id. Use semantic meaning, not wording overlap. Do not compare the parent and HLG' with each other yet. Do not make the final branch decision.

If the documentation clearly expresses the branch's functional WHY-level intention, summarize it in one concise sentence as documented_branch_intention; otherwise use null.

# Output
Return JSON only:
{
  "rationale": "max 2 concise sentences",
  "parent_supported": true | false,
  "reconstructed_supported": true | false,
  "documented_branch_intention": "one concise sentence or null"
}"""


def _build_branch_grounding_user_prompt(
    branch_id: str,
    project_description: str,
    reconstructed_high_level_goal: BottomUpHighLevelGoal,
    existing_high_level_goals: dict[str, HighLevelGoal],
) -> str:
    return f"""<documentation>
{project_description}
</documentation>

<current_hlgs>
{_format_existing_high_level_goals_block(existing_high_level_goals)}
</current_hlgs>

<current_branch_id>{branch_id}</current_branch_id>

<reconstructed_hlg>
{reconstructed_high_level_goal.reconstructed_high_level_goal}
</reconstructed_hlg>

Evaluate documentation grounding only."""


def _build_branch_alignment_system_prompt() -> str:
    """Stage 2: compare the parent HLG and HLG' at WHY level."""
    return _GORE_EXPERT_PREAMBLE + """# Stage 2 - Round-trip alignment
Assess only whether HLG' preserves the same core functional intention as the parent HLG.

Preservation means the same actor/stakeholder perspective, WHY-level objective, functional scope, and intended outcome. Different wording and extra documented detail are acceptable when the parent's core intention is still preserved.

Do not decide whether another project-wide HLG is missing. Do not prepare regeneration text.

# Output
Return JSON only:
{
  "rationale": "max 2 concise sentences",
  "preserves_parent_intention": true | false
}"""


def _build_branch_alignment_user_prompt(
    original_high_level_goal: HighLevelGoal,
    reconstructed_high_level_goal: BottomUpHighLevelGoal,
    grounding: BranchGroundingLLMOutput,
) -> str:
    grounding_block = json.dumps(grounding.model_dump(mode="json"), ensure_ascii=False)
    return f"""<parent_hlg>
name: {original_high_level_goal.name}
description: {original_high_level_goal.description}
actor: {original_high_level_goal.actor.name}
</parent_hlg>

<reconstructed_hlg>
{reconstructed_high_level_goal.reconstructed_high_level_goal}
</reconstructed_hlg>

<documentation_grounding>
{grounding_block}
</documentation_grounding>

Evaluate semantic round-trip alignment only."""


def _build_branch_regeneration_system_prompt() -> str:
    """Stage 3: prepare the focused replacement-HLG generation input."""
    return _GORE_EXPERT_PREAMBLE + """# Stage 3 - Regeneration preparation
The branch failed documentation grounding and/or round-trip alignment. Return REGENERATE_HIGH_LEVEL_GOAL and prepare the focused input for the original HLG generator.

Do not generate the final HLG. Do not add a new project-wide HLG here.

""" + _GENERATION_DESCRIPTION_RULES + """# Output
Return JSON only:
{
  "rationale": "max 2 concise sentences",
  "decision": "REGENERATE_HIGH_LEVEL_GOAL",
  "generation_project_description": "1-3 concise stakeholder-style sentences"
}"""


def _build_branch_regeneration_user_prompt(
    project_description: str,
    original_high_level_goal: HighLevelGoal,
    reconstructed_high_level_goal: BottomUpHighLevelGoal,
    grounding: BranchGroundingLLMOutput,
    alignment: BranchAlignmentLLMOutput,
) -> str:
    grounding_block = json.dumps(grounding.model_dump(mode="json"), ensure_ascii=False)
    alignment_block = json.dumps(alignment.model_dump(mode="json"), ensure_ascii=False)
    return f"""<documentation>
{project_description}
</documentation>

<parent_hlg>
name: {original_high_level_goal.name}
description: {original_high_level_goal.description}
actor: {original_high_level_goal.actor.name}
</parent_hlg>

<reconstructed_hlg>
{reconstructed_high_level_goal.reconstructed_high_level_goal}
</reconstructed_hlg>

<stage_1>
{grounding_block}
</stage_1>

<stage_2>
{alignment_block}
</stage_2>

Prepare only the focused replacement-HLG generation input."""

_StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)
_StageResultT = TypeVar("_StageResultT")


def _parse_structured_llm_output(
    raw_response: str,
    output_model: type[_StructuredOutputT],
    stage_label: str,
) -> _StructuredOutputT:
    """Parse and validate JSON returned by one evaluator LLM stage."""
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
            f"Llama returned JSON that does not respect "
            f"{output_model.__name__} for the {stage_label} stage. "
            f"Response received: {raw_response}"
        ) from exc


def _run_evaluation_stage(
    user_prompt: str,
    system_prompt: str,
    output_model: type[_StructuredOutputT],
    stage_label: str,
    process: Callable[[_StructuredOutputT], _StageResultT],
) -> _StageResultT:
    """Run one structured evaluator call with bounded semantic retries."""
    retry_feedback = ""
    last_error: GlobalGoalEvaluationError | None = None

    for _ in range(MAX_SEMANTIC_RETRIES + 1):
        prompt = (
            user_prompt
            if not retry_feedback
            else f"{user_prompt}\n\n{retry_feedback}"
        )
        raw_output = generate_response_llama(prompt, system_prompt)

        try:
            parsed_output = _parse_structured_llm_output(
                raw_output,
                output_model,
                stage_label,
            )
            return process(parsed_output)
        except GlobalGoalEvaluationError as exc:
            last_error = exc
            error_summary = str(exc).replace("\n", " ")[:400]
            retry_feedback = (
                f"Previous output was invalid: {error_summary}. "
                "Return corrected JSON only."
            )

    if last_error is None:
        raise GlobalGoalEvaluationError(
            f"The {stage_label} stage failed without a recoverable result."
        )
    raise last_error


def _build_branch_generation_request(
    branch_id: str,
    llm_output: GlobalGoalEvaluationLLMOutput,
    original_high_level_goal: HighLevelGoal,
) -> HighLevelGoalGenerationRequest:
    """Create the replacement-HLG request emitted by an unstable branch."""
    if (
        llm_output.decision
        != GlobalGoalEvaluationDecision.REGENERATE_HIGH_LEVEL_GOAL
    ):
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': only REGENERATE_HIGH_LEVEL_GOAL may create "
            "a branch-level HLG generation request."
        )

    generation_description = llm_output.generation_project_description
    if not generation_description or not generation_description.strip():
        raise GlobalGoalEvaluationError(
            f"Branch '{branch_id}': REGENERATE_HIGH_LEVEL_GOAL requires a "
            "non-empty generation_project_description."
        )

    return HighLevelGoalGenerationRequest(
        request_id=f"{branch_id}_regenerate_hlg",
        action=HighLevelGoalGenerationAction.REPLACE_EXISTING_HIGH_LEVEL_GOAL,
        source=HighLevelGoalGenerationSource.BRANCH_EVALUATION,
        generator_input=HighLevelGoalGeneratorInput(
            project_description=generation_description,
            actors=Actors(actors=[original_high_level_goal.actor]),
        ),
        target_branch_id=branch_id,
        origin_branch_id=branch_id,
        generator_guidance=llm_output.rationale,
        rationale=llm_output.rationale,
    )


def _validate_branch_decision(
    output: GlobalGoalEvaluationLLMOutput,
) -> GlobalGoalEvaluationLLMOutput:
    """Reject every branch decision outside the two-way local gate."""
    if output.decision not in _BRANCH_ALLOWED_DECISIONS:
        raise GlobalGoalEvaluationError(
            "Branch evaluation supports only CONFIRM_BRANCH or "
            "REGENERATE_HIGH_LEVEL_GOAL; received "
            f"'{output.decision.value}'."
        )
    if output.matched_branch_id is not None:
        raise GlobalGoalEvaluationError(
            "matched_branch_id must always be null in the simplified branch "
            "evaluation."
        )
    return output


def evaluate_branch(
    branch_id: str,
    project_description: str,
    reconstructed_high_level_goal: BottomUpHighLevelGoal,
    existing_high_level_goals: dict[str, HighLevelGoal],
) -> GlobalGoalEvaluationResult:
    """Evaluate one non-empty branch through up to three focused LLM stages.

    Stage 1 grounds parent and HLG' in the documentation. Stage 2 checks
    round-trip semantic preservation. A branch is confirmed only when both
    goals are documentation-supported and HLG' preserves the parent intention.
    Stage 3 runs only for an unstable branch and prepares the focused HLG
    replacement request. The only final decisions remain CONFIRM_BRANCH and
    REGENERATE_HIGH_LEVEL_GOAL.
    """
    original_high_level_goal = existing_high_level_goals.get(branch_id)
    if original_high_level_goal is None:
        raise GlobalGoalEvaluationError(
            f"Unknown current branch_id '{branch_id}'."
        )

    # Stage 1 - Documentation grounding.
    grounding = _run_evaluation_stage(
        user_prompt=_build_branch_grounding_user_prompt(
            branch_id=branch_id,
            project_description=project_description,
            reconstructed_high_level_goal=reconstructed_high_level_goal,
            existing_high_level_goals=existing_high_level_goals,
        ),
        system_prompt=_build_branch_grounding_system_prompt(),
        output_model=BranchGroundingLLMOutput,
        stage_label="branch documentation grounding",
        process=lambda output: output,
    )

    # Stage 2 - Parent/HLG' semantic alignment.
    alignment = _run_evaluation_stage(
        user_prompt=_build_branch_alignment_user_prompt(
            original_high_level_goal=original_high_level_goal,
            reconstructed_high_level_goal=reconstructed_high_level_goal,
            grounding=grounding,
        ),
        system_prompt=_build_branch_alignment_system_prompt(),
        output_model=BranchAlignmentLLMOutput,
        stage_label="branch round-trip alignment",
        process=lambda output: output,
    )

    branch_confirmed = (
        grounding.parent_supported
        and grounding.reconstructed_supported
        and alignment.preserves_parent_intention
    )

    if branch_confirmed:
        decision = GlobalGoalEvaluationDecision.CONFIRM_BRANCH
        rationale = (
            f"Grounding: {grounding.rationale} "
            f"Alignment: {alignment.rationale}"
        ).strip()
        return GlobalGoalEvaluationResult(
            branch_id=branch_id,
            decision=decision,
            original_high_level_goal=original_high_level_goal,
            reconstructed_high_level_goal=(
                reconstructed_high_level_goal.reconstructed_high_level_goal
            ),
            matched_high_level_goal=None,
            generation_request=None,
            rationale=rationale,
            operational_action=_OPERATIONAL_ACTION_BY_DECISION[decision],
            requires_high_level_regeneration=False,
            requires_low_level_regeneration=False,
        )

    # Stage 3 - Regeneration preparation. It runs only when the branch failed
    # grounding and/or alignment, keeping the normal case to two short calls.
    regeneration_output = _run_evaluation_stage(
        user_prompt=_build_branch_regeneration_user_prompt(
            project_description=project_description,
            original_high_level_goal=original_high_level_goal,
            reconstructed_high_level_goal=reconstructed_high_level_goal,
            grounding=grounding,
            alignment=alignment,
        ),
        system_prompt=_build_branch_regeneration_system_prompt(),
        output_model=GlobalGoalEvaluationLLMOutput,
        stage_label="branch HLG regeneration preparation",
        process=_validate_branch_decision,
    )

    if (
        regeneration_output.decision
        != GlobalGoalEvaluationDecision.REGENERATE_HIGH_LEVEL_GOAL
    ):
        raise GlobalGoalEvaluationError(
            "Stage 3 must return REGENERATE_HIGH_LEVEL_GOAL for an unstable "
            "branch."
        )

    generation_request = _build_branch_generation_request(
        branch_id=branch_id,
        llm_output=regeneration_output,
        original_high_level_goal=original_high_level_goal,
    )
    decision = GlobalGoalEvaluationDecision.REGENERATE_HIGH_LEVEL_GOAL

    return GlobalGoalEvaluationResult(
        branch_id=branch_id,
        decision=decision,
        original_high_level_goal=original_high_level_goal,
        reconstructed_high_level_goal=(
            reconstructed_high_level_goal.reconstructed_high_level_goal
        ),
        matched_high_level_goal=None,
        generation_request=generation_request,
        rationale=regeneration_output.rationale,
        operational_action=_OPERATIONAL_ACTION_BY_DECISION[decision],
        requires_high_level_regeneration=True,
        requires_low_level_regeneration=True,
    )


def _build_empty_branch_system_prompt() -> str:
    """Build the prompt used when a branch has no LLG decomposition."""
    return _GORE_EXPERT_PREAMBLE + """# Task
The current branch has no LLGs, so no HLG' can be reconstructed and the branch cannot be confirmed. Return REGENERATE_HIGH_LEVEL_GOAL and prepare a focused input for the original HLG generator. The replacement HLG will later receive a new LLG decomposition.

""" + _GENERATION_DESCRIPTION_RULES + """# Output
Return JSON only, with exactly these fields:
{
  "rationale": "max 2 concise sentences",
  "decision": "REGENERATE_HIGH_LEVEL_GOAL",
  "generation_project_description": "string"
}"""

def _build_empty_branch_user_prompt(
    branch_id: str,
    project_description: str,
    existing_high_level_goals: dict[str, HighLevelGoal],
) -> str:
    return f"""<documentation>
{project_description}
</documentation>

<current_hlgs>
{_format_existing_high_level_goals_block(existing_high_level_goals)}
</current_hlgs>

<current_branch_id>
{branch_id}
</current_branch_id>

Prepare the regeneration input for this empty branch."""

def evaluate_empty_branch(
    branch_id: str,
    project_description: str,
    existing_high_level_goals: dict[str, HighLevelGoal],
) -> GlobalGoalEvaluationResult:
    """Regenerate an empty branch from HLG level, then generate its LLGs."""
    original_high_level_goal = existing_high_level_goals.get(branch_id)
    if original_high_level_goal is None:
        raise GlobalGoalEvaluationError(
            f"Unknown empty branch_id '{branch_id}'."
        )

    def _process(output: GlobalGoalEvaluationLLMOutput):
        output = _validate_branch_decision(output)
        if output.decision != GlobalGoalEvaluationDecision.REGENERATE_HIGH_LEVEL_GOAL:
            raise GlobalGoalEvaluationError(
                "An empty branch must return REGENERATE_HIGH_LEVEL_GOAL."
            )
        return output

    llm_output = _run_evaluation_stage(
        user_prompt=_build_empty_branch_user_prompt(
            branch_id=branch_id,
            project_description=project_description,
            existing_high_level_goals=existing_high_level_goals,
        ),
        system_prompt=_build_empty_branch_system_prompt(),
        output_model=GlobalGoalEvaluationLLMOutput,
        stage_label="empty-branch regeneration",
        process=_process,
    )

    generation_request = _build_branch_generation_request(
        branch_id=branch_id,
        llm_output=llm_output,
        original_high_level_goal=original_high_level_goal,
    )

    decision = GlobalGoalEvaluationDecision.REGENERATE_HIGH_LEVEL_GOAL
    return GlobalGoalEvaluationResult(
        branch_id=branch_id,
        decision=decision,
        original_high_level_goal=original_high_level_goal,
        reconstructed_high_level_goal=None,
        matched_high_level_goal=None,
        generation_request=generation_request,
        rationale=llm_output.rationale,
        operational_action=_OPERATIONAL_ACTION_BY_DECISION[decision],
        requires_high_level_regeneration=True,
        requires_low_level_regeneration=True,
    )


def evaluate_all_branches(
    project_description: str,
    existing_high_level_goals: dict[str, HighLevelGoal],
    reconstructed_high_level_goals: dict[str, BottomUpHighLevelGoal],
    empty_branches: list[str],
    branch_low_level_goals: object | None = None,
) -> tuple[dict[str, GlobalGoalEvaluationResult], dict[str, str]]:
    """Produce exactly one two-way evaluation for every current HLG branch.

    ``branch_low_level_goals`` is accepted only for caller compatibility and is
    deliberately ignored: individual LLGs are not input to this evaluator.
    """
    del branch_low_level_goals
    results: dict[str, GlobalGoalEvaluationResult] = {}
    errors: dict[str, str] = {}

    expected_branch_ids = set(existing_high_level_goals)
    reconstructed_branch_ids = set(reconstructed_high_level_goals)
    empty_branch_ids = set(empty_branches)

    for branch_id in sorted(reconstructed_branch_ids - expected_branch_ids):
        errors[branch_id] = (
            "GlobalGoalEvaluationError: a bottom-up reconstruction was "
            "provided for an unexpected branch."
        )
    for branch_id in sorted(empty_branch_ids - expected_branch_ids):
        errors[branch_id] = (
            "GlobalGoalEvaluationError: an unknown branch was marked empty."
        )

    overlap = reconstructed_branch_ids & empty_branch_ids
    for branch_id in sorted(overlap):
        errors[branch_id] = (
            "GlobalGoalEvaluationError: a branch cannot simultaneously have "
            "a reconstructed HLG and be marked empty."
        )

    for branch_id in existing_high_level_goals:
        if branch_id in overlap:
            continue

        reconstructed = reconstructed_high_level_goals.get(branch_id)
        if reconstructed is not None:
            try:
                results[branch_id] = evaluate_branch(
                    branch_id=branch_id,
                    project_description=project_description,
                    reconstructed_high_level_goal=reconstructed,
                    existing_high_level_goals=existing_high_level_goals,
                )
            except Exception as exc:
                errors[branch_id] = f"{type(exc).__name__}: {exc}"
            continue

        if branch_id in empty_branch_ids:
            try:
                results[branch_id] = evaluate_empty_branch(
                    branch_id=branch_id,
                    project_description=project_description,
                    existing_high_level_goals=existing_high_level_goals,
                )
            except Exception as exc:
                errors[branch_id] = f"{type(exc).__name__}: {exc}"
            continue

        errors[branch_id] = (
            "GlobalGoalEvaluationError: expected branch has neither a "
            "reconstructed HLG nor an empty-branch marker."
        )

    return results, errors


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
            "GlobalGoalEvaluationError: dictionary key does not match "
            "evaluation.branch_id.",
        )

    is_complete = (
        not completeness_errors
        and not missing_branch_ids
        and not unexpected_branch_ids
        and not inconsistent_branch_ids
        and expected_set == evaluated_set
    )

    payload = {
        "schema_version": "4.0",
        "status": (
            "READY_FOR_ORCHESTRATION" if is_complete else "INCOMPLETE_EVALUATION"
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Global documentation coverage evaluator
# ---------------------------------------------------------------------------


class DocumentationCoverageEvaluationError(ValueError):
    """Raised when the documentation-coverage response is invalid."""


def _build_documentation_coverage_system_prompt() -> str:
    """Build a concise global HLG-coverage detector prompt."""
    return _GORE_EXPERT_PREAMBLE + """# Task
Determine whether the current HLG set covers every autonomous documented functional WHY-level intention. The documentation is the source of truth; current HLGs are the coverage baseline; confirmed bottom-up HLG' values are discovery cues only.

Inspect HLG' cues for documented scope that may have emerged from the LLGs but is not represented by any current HLG. Keep a missing intention only when all conditions hold: it is supported by the documentation, autonomous at WHY level, semantically absent from the current HLG set, and not merely an operation, refinement, UI detail, implementation mechanism, field, channel, or sub-capability.

Return COMPLETE when no such intention exists. Return MISSING_HIGH_LEVEL_GOALS otherwise. For each missing intention provide a concise stakeholder-style project_description, its responsible actor, short source_evidence, and a brief rationale. Use an existing actor when it represents the documented role; introduce a different actor only when the documentation clearly requires one.

# Output
Return JSON only:
{
  "status": "COMPLETE | MISSING_HIGH_LEVEL_GOALS",
  "missing_high_level_goals": [
    {
      "project_description": "string",
      "actor": {"name": "string", "description": "string"},
      "source_evidence": "string",
      "rationale": "string"
    }
  ],
  "observations": ["string"]
}
For COMPLETE, missing_high_level_goals must be empty."""

def _build_documentation_coverage_user_prompt(
    project_description: str,
    current_high_level_goals: HighLevelGoals,
    reconstructed_high_level_goals: dict[str, BottomUpHighLevelGoal] | None,
) -> str:
    """Provide the coverage evidence, then ask for the coverage decision."""
    goals_block = "\n".join(
        f"- [{index}] {goal.name}: {goal.description} "
        f"(actor: {goal.actor.name})"
        for index, goal in enumerate(current_high_level_goals.goals, start=1)
    )
    reconstructed_block = _format_reconstructed_high_level_goals_block(
        reconstructed_high_level_goals
    )

    return f"""<documentation>
{project_description}
</documentation>

<current_hlgs>
{goals_block}
</current_hlgs>

<confirmed_reconstructed_hlgs>
{reconstructed_block}
</confirmed_reconstructed_hlgs>

Using the documentation as evidence, decide whether any autonomous WHY-level intention is still missing from current_hlgs. Treat confirmed_reconstructed_hlgs only as cues."""

def _build_documentation_coverage_selection_system_prompt() -> str:
    """Build a concise precision-gate prompt for coverage proposals."""
    return _GORE_EXPERT_PREAMBLE + """# Task
Select the smallest defensible subset of candidate missing HLG intentions. Keep a proposal only when it is documented, autonomous at WHY level, not semantically covered by a current HLG, and not redundant with another selected proposal. Reject operational details, refinements, benefits, implementation mechanisms, and weakly grounded proposals. When uncertain, reject.

Use source_evidence only as a pointer: verify it against the full documentation. Treat actor aliases as the same role when their functional responsibility is equivalent. Select existing proposal ids only; do not rewrite or create proposals.

# Output
Return JSON only:
{"selected_proposal_ids": ["proposal_XXX"], "rationale": "brief selection rationale"}"""

def _build_documentation_coverage_selection_user_prompt(
    project_description: str,
    current_high_level_goals: HighLevelGoals,
    reconstructed_high_level_goals: dict[str, BottomUpHighLevelGoal] | None,
    proposals: dict[str, object],
) -> str:
    """Provide only the evidence needed by the proposal precision gate."""
    # HLG' values were already consumed by the first coverage detector to form
    # these proposals. Repeating them here adds tokens without adding evidence.
    del reconstructed_high_level_goals

    goals_block = "\n".join(
        f"- [{index}] {goal.name}: {goal.description} "
        f"(actor: {goal.actor.name})"
        for index, goal in enumerate(current_high_level_goals.goals, start=1)
    )
    proposal_blocks = []
    for proposal_id, proposal in proposals.items():
        proposal_blocks.append(
            "\n".join(
                [
                    f"id: {proposal_id}",
                    f"actor: {proposal.actor.name}",
                    f"intention: {proposal.project_description}",
                    f"evidence: {proposal.source_evidence}",
                ]
            )
        )
    proposals_block = "\n\n".join(proposal_blocks)

    return f"""<documentation>
{project_description}
</documentation>

<current_hlgs>
{goals_block}
</current_hlgs>

<candidate_proposals>
{proposals_block}
</candidate_proposals>

Select the minimal grounded subset of candidate_proposals."""

def _select_minimal_documentation_coverage_proposals(
    project_description: str,
    current_high_level_goals: HighLevelGoals,
    reconstructed_high_level_goals: dict[str, BottomUpHighLevelGoal] | None,
    proposals: list,
) -> tuple[list, str]:
    """Verify coverage proposals as a set before any HLG generation request."""
    if not proposals:
        return [], "No coverage proposals were supplied."

    proposals_by_id = {
        f"proposal_{index + 1:03d}": proposal
        for index, proposal in enumerate(proposals)
    }
    valid_ids = set(proposals_by_id)

    def _validate_selection(
        output: DocumentationCoverageSelectionLLMOutput,
    ) -> tuple[list, str]:
        unknown_ids = [
            proposal_id
            for proposal_id in output.selected_proposal_ids
            if proposal_id not in valid_ids
        ]
        if unknown_ids:
            raise GlobalGoalEvaluationError(
                "selected_proposal_ids contains unknown ids "
                f"{sorted(unknown_ids)}; valid ids are {sorted(valid_ids)}."
            )
        selected = [
            proposals_by_id[proposal_id]
            for proposal_id in output.selected_proposal_ids
        ]
        return selected, output.rationale

    return _run_evaluation_stage(
        user_prompt=_build_documentation_coverage_selection_user_prompt(
            project_description=project_description,
            current_high_level_goals=current_high_level_goals,
            reconstructed_high_level_goals=reconstructed_high_level_goals,
            proposals=proposals_by_id,
        ),
        system_prompt=_build_documentation_coverage_selection_system_prompt(),
        output_model=DocumentationCoverageSelectionLLMOutput,
        stage_label="documentation coverage proposal verification",
        process=_validate_selection,
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
    reconstructed_high_level_goals: dict[str, BottomUpHighLevelGoal] | None = None,
) -> DocumentationCoverageResult:
    """
    Evaluate global HLG coverage and produce normal top-down generation
    requests for genuinely missing functional intentions.
    """
    base_user_prompt = _build_documentation_coverage_user_prompt(
        project_description=project_description,
        current_high_level_goals=current_high_level_goals,
        reconstructed_high_level_goals=reconstructed_high_level_goals,
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

            proposals_to_check = list(output.missing_high_level_goals)
            if (
                output.status
                == DocumentationCoverageStatus.MISSING_HIGH_LEVEL_GOALS
            ):
                (
                    proposals_to_check,
                    proposal_selection_rationale,
                ) = _select_minimal_documentation_coverage_proposals(
                    project_description=project_description,
                    current_high_level_goals=current_high_level_goals,
                    reconstructed_high_level_goals=reconstructed_high_level_goals,
                    proposals=proposals_to_check,
                )
                if not proposals_to_check:
                    raise DocumentationCoverageEvaluationError(
                        "The coverage detector proposed missing HLGs, but the "
                        "independent proposal verifier rejected every proposal "
                        "as covered, redundant, too operational, or "
                        "insufficiently grounded."
                    )
            else:
                proposal_selection_rationale = (
                    "Coverage detector returned COMPLETE; no proposal "
                    "verification was necessary."
                )

            seen_intentions: set[tuple[str, str]] = set()
            retained_proposals = []
            generation_requests: list[HighLevelGoalGenerationRequest] = []
            coverage_observations = list(output.observations)
            coverage_observations.append(
                "Coverage proposal verifier: "
                f"{proposal_selection_rationale}"
            )

            for proposal in proposals_to_check:
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
                        generator_guidance=(
                            f"{proposal.rationale} Source evidence: "
                            f"{proposal.source_evidence}"
                        ),
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
            error_summary = str(exc).replace("\n", " ")[:400]
            retry_feedback = (
                f"Previous output was invalid: {error_summary}. "
                "Return corrected JSON only."
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
