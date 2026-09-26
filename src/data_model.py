"""Shared models for the original top-down and bottom-up pipelines."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class DocumentDescription(BaseModel):
    description: str


class Actor(BaseModel):
    name: str
    description: str


class Actors(BaseModel):
    actors: list[Actor]


class HighLevelGoal(BaseModel):
    name: str
    description: str
    actor: Actor


class HighLevelGoals(BaseModel):
    goals: list[HighLevelGoal]


class LowLevelGoal(BaseModel):
    name: str
    description: str
    high_level_associated: HighLevelGoal


class LowLevelGoals(BaseModel):
    low_level_goals: list[LowLevelGoal]


class API(BaseModel):
    api_name: str
    api_path: str
    description: str
    request_type: str


class APIMapping(BaseModel):
    APIs: list[API]
    low_level_goal: LowLevelGoal


class Critique(BaseModel):
    score: float
    comment: str


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class GoalBranch(BaseModel):
    branch_id: str
    high_level_goal: HighLevelGoal
    low_level_goals: list[LowLevelGoal]


class BottomUpHighLevelGoalLLMOutput(BaseModel):
    reconstructed_high_level_goal: str
    rationale: str


class BottomUpHighLevelGoal(BottomUpHighLevelGoalLLMOutput):
    branch_id: str
    source_low_level_goal_ids: list[str]


class HighLevelGoalDecision(str, Enum):
    KEEP = "KEEP_ORIGINAL_HIGH_LEVEL_GOAL"
    REWRITE = "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL"
    REMOVE = "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL"


class HighLevelGoalEvaluation(BaseModel):
    rationale: str
    decision: HighLevelGoalDecision
    rewriting_focus: str | None = None
    confidence: ConfidenceLevel

    @model_validator(mode="after")
    def check_rewriting_focus(self):
        if self.decision == HighLevelGoalDecision.REWRITE and not self.rewriting_focus:
            raise ValueError("REWRITE_ORIGINAL_HIGH_LEVEL_GOAL requires rewriting_focus")
        return self


class HighLevelGoalReplacementRequest(BaseModel):
    rationale: str
    generation_project_description: str
    actor: Actor


class LowLevelGoalDecision(str, Enum):
    KEEP = "KEEP_LOW_LEVEL_GOALS"
    REGENERATE = "REGENERATE_LOW_LEVEL_GOALS"


class LowLevelGoalEvaluation(BaseModel):
    rationale: str
    decision: LowLevelGoalDecision
    regeneration_feedback: str | None = None
    unsupported_or_misleading_llg_ids: list[str] = Field(default_factory=list)
    missing_essential_capabilities: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel

    @field_validator("unsupported_or_misleading_llg_ids", mode="before")
    @classmethod
    def coerce_llg_ids_to_strings(cls, value):
        """Accept numeric IDs occasionally emitted by the evaluator JSON."""
        if value is None:
            return []
        if isinstance(value, (str, int, float)):
            value = [value]
        return [str(item) for item in value]


class MissingHighLevelGoalDecision(str, Enum):
    NO_MISSING = "NO_MISSING_HIGH_LEVEL_GOALS"
    FOUND = "MISSING_HIGH_LEVEL_GOALS_FOUND"


class MissingHighLevelGoalRequest(BaseModel):
    actor: str
    generation_project_description: str


class MissingHighLevelGoalEvaluation(BaseModel):
    decision: MissingHighLevelGoalDecision
    rationale: str
    missing_goal_requests: list[MissingHighLevelGoalRequest] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_requests(self):
        if self.decision == MissingHighLevelGoalDecision.FOUND and not self.missing_goal_requests:
            raise ValueError(
                "MISSING_HIGH_LEVEL_GOALS_FOUND requires at least one request."
            )
        if self.decision == MissingHighLevelGoalDecision.NO_MISSING and self.missing_goal_requests:
            raise ValueError(
                "NO_MISSING_HIGH_LEVEL_GOALS cannot contain requests."
            )
        return self


class GlobalGoalEvaluationDecision(str, Enum):
    CONFIRM_BRANCH = "CONFIRM_BRANCH"
    EVALUATION_INCONCLUSIVE = "EVALUATION_INCONCLUSIVE"
    REGENERATE_LOW_LEVEL_GOALS = "REGENERATE_LOW_LEVEL_GOALS"
    LLG_REGENERATION_LIMIT_REACHED = "LLG_REGENERATION_LIMIT_REACHED"
    DISCOVER_NEW_HIGH_LEVEL_GOAL = "DISCOVER_NEW_HIGH_LEVEL_GOAL"
    REWRITE_ORIGINAL_HIGH_LEVEL_GOAL = "REWRITE_ORIGINAL_HIGH_LEVEL_GOAL"
    REMOVE_ORIGINAL_HIGH_LEVEL_GOAL = "REMOVE_ORIGINAL_HIGH_LEVEL_GOAL"
    CHECK_MISSING_HIGH_LEVEL_GOALS = "CHECK_MISSING_HIGH_LEVEL_GOALS"
    MISSING_HIGH_LEVEL_GOALS_FOUND = "MISSING_HIGH_LEVEL_GOALS_FOUND"
    NO_MISSING_HIGH_LEVEL_GOALS = "NO_MISSING_HIGH_LEVEL_GOALS"


class GlobalGoalEvaluationResult(BaseModel):
    branch_id: str
    rationale: str
    final_decision: GlobalGoalEvaluationDecision
    high_level_evaluation: HighLevelGoalEvaluation
    replacement_request: HighLevelGoalReplacementRequest | None = None
    low_level_evaluation: LowLevelGoalEvaluation | None = None


class HighLevelGoalGenerationAction(str, Enum):
    ADD_NEW_HIGH_LEVEL_GOAL = "ADD_NEW_HIGH_LEVEL_GOAL"
    REPLACE_EXISTING_HIGH_LEVEL_GOAL = "REPLACE_EXISTING_HIGH_LEVEL_GOAL"


class HighLevelGoalGeneratorInput(BaseModel):
    project_description: str
    actors: Actors


class HighLevelGoalGenerationRequest(BaseModel):
    request_id: str
    action: HighLevelGoalGenerationAction
    generator_input: HighLevelGoalGeneratorInput
    rationale: str
    generator_guidance: str | None = None
    target_branch_id: str | None = None


class LowLevelGoalRegenerationRequest(BaseModel):
    high_level_goals: HighLevelGoals
    guidance_by_parent_name: dict[str, str] = Field(default_factory=dict)
    existing_low_level_goals: LowLevelGoals = Field(
        default_factory=lambda: LowLevelGoals(low_level_goals=[])
    )
    max_goals_by_parent_name: dict[str, int] = Field(default_factory=dict)


class GlobalGoalCycleStopReason(str, Enum):
    ALL_BRANCHES_CONFIRMED = "ALL_BRANCHES_CONFIRMED"
    NO_CHANGES_APPLIED = "NO_CHANGES_APPLIED"
    NO_ACTIONS_REMAIN_WITH_WARNINGS = "NO_ACTIONS_REMAIN_WITH_WARNINGS"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"


class GlobalGoalCycleIteration(BaseModel):
    iteration: int
    reconstructions: dict[str, BottomUpHighLevelGoal]
    evaluations: dict[str, GlobalGoalEvaluationResult]
    high_level_goals: HighLevelGoals
    low_level_goals: LowLevelGoals
    missing_hlg_evaluation: MissingHighLevelGoalEvaluation | None = None
    missing_hlg_decision: GlobalGoalEvaluationDecision | None = None
    llg_regeneration_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class LowLevelGoalCleanupBranch(BaseModel):
    branch_id: str
    high_level_goal: HighLevelGoal
    evaluation: LowLevelGoalEvaluation
    removed_low_level_goals: list[LowLevelGoal] = Field(default_factory=list)
    retained_low_level_goal_count: int
    applied: bool = False
    rationale: str


class LowLevelGoalCleanupPrepass(BaseModel):
    branches: dict[str, LowLevelGoalCleanupBranch] = Field(default_factory=dict)
    initial_low_level_goal_count: int
    final_low_level_goal_count: int
    low_level_goals: LowLevelGoals
    warnings: list[str] = Field(default_factory=list)


class GlobalGoalCycleResult(BaseModel):
    converged: bool
    stop_reason: GlobalGoalCycleStopReason
    completed_iterations: int
    final_high_level_goals: HighLevelGoals
    final_low_level_goals: LowLevelGoals
    llg_cleanup_prepass: LowLevelGoalCleanupPrepass | None = None
    iterations: list[GlobalGoalCycleIteration] = Field(default_factory=list)
    llg_regeneration_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
