"""Reconstruct one diagnostic HLG from the LLGs of a branch."""

from src.data_model import (
    BottomUpHighLevelGoal,
    BottomUpHighLevelGoalLLMOutput,
    GoalBranch,
)
from src.llm_clients import generate_response


SYSTEM_PROMPT = """You are an expert in Goal-Oriented Requirements Engineering.
Infer one High-Level Goal from the supplied Low-Level Goals. Explain WHY the
operations belong together. Do not invent functionality and do not mention the
hidden parent goal. Return only the requested JSON."""


def reconstruct_high_level_goal(branch: GoalBranch) -> BottomUpHighLevelGoal:
    if not branch.low_level_goals:
        return BottomUpHighLevelGoal(
            branch_id=branch.branch_id,
            reconstructed_high_level_goal=(
                "No High-Level Goal can be reconstructed because this branch "
                "has no Low-Level Goals."
            ),
            rationale=(
                "The empty decomposition is a material coverage gap. Evaluate "
                "the original HLG normally; if it is valid, generate its "
                "required Low-Level Goals instead of removing it merely because "
                "the branch is empty."
            ),
            source_low_level_goal_ids=[],
        )

    lines = [
        f"- llg_{index:03d}: {goal.description}"
        for index, goal in enumerate(branch.low_level_goals, start=1)
    ]
    prompt = """Given these Low-Level Goals, reconstruct the single High-Level
Goal that explains why they should be pursued together.

Low-Level Goals:
{goals}

Output only valid JSON:
{{
  "reconstructed_high_level_goal": "...",
  "rationale": "Brief explanation of the inferred intention."
}}""".format(goals="\n".join(lines))

    output = generate_response(prompt, SYSTEM_PROMPT, BottomUpHighLevelGoalLLMOutput)
    return BottomUpHighLevelGoal(
        branch_id=branch.branch_id,
        reconstructed_high_level_goal=output.reconstructed_high_level_goal,
        rationale=output.rationale,
        source_low_level_goal_ids=[goal.name for goal in branch.low_level_goals],
    )


def reconstruct_all_branches(branches: list[GoalBranch]):
    return {branch.branch_id: reconstruct_high_level_goal(branch) for branch in branches}
