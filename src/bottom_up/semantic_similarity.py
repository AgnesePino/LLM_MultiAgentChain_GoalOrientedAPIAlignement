"""
Shared semantic-similarity utilities for the bottom-up feedback loop.

The module supports two distinct checks:
- semantic duplicate detection for High-Level Goals (HLGs);
- semantic repeated-state detection for HLG/LLG collections.

Embedding and cosine-similarity computation is delegated to
``src.evaluation.goal_evaluator.GoalEvaluator``. The state-equivalence logic
adds a local one-to-one Hungarian matching criterion because that policy is
specific to repeated-state detection.

The thresholds below are configurable defaults and must be calibrated
empirically on the thesis datasets. Duplicate detection and repeated-state
detection intentionally use separate defaults because they answer different
semantic questions.
"""

from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.bottom_up.goal_reconstructor import normalize_goal_name
from src.data_model import (
    HighLevelGoal,
    HighLevelGoals,
    LowLevelGoal,
    LowLevelGoals,
)


DEFAULT_HLG_DUPLICATE_SIMILARITY_THRESHOLD = 0.90
DEFAULT_STATE_SIMILARITY_THRESHOLD = 0.92


@lru_cache(maxsize=1)
def _get_goal_evaluator():
    """Lazily create and reuse the embedding-based GoalEvaluator."""
    from src.evaluation.goal_evaluator import GoalEvaluator

    return GoalEvaluator()


def _high_level_goal_duplicate_text(goal: HighLevelGoal) -> str:
    """HLG text used for duplicate detection within an already matched actor."""
    return f"{goal.name}. {goal.description}"


def _high_level_goal_state_text(goal: HighLevelGoal) -> str:
    """HLG text used when comparing complete cycle states."""
    return f"{goal.actor.name}. {goal.name}. {goal.description}"


def _low_level_goal_state_text(goal: LowLevelGoal) -> str:
    """LLG text used for state comparison, including its parent context."""
    parent = goal.high_level_associated
    return (
        f"{parent.actor.name}. {parent.name}. {parent.description}. "
        f"{goal.name}. {goal.description}"
    )


def find_semantic_duplicate_high_level_goal(
    candidate: HighLevelGoal,
    existing_goals: Sequence[HighLevelGoal],
    threshold: float = DEFAULT_HLG_DUPLICATE_SIMILARITY_THRESHOLD,
) -> tuple[HighLevelGoal | None, float | None]:
    """
    Return the best same-actor semantic duplicate and its similarity score.

    Actor equality is checked deterministically before embedding comparison,
    so the embedding text contains only the HLG name and description. This
    avoids inflating similarity simply because compared goals share the same
    actor.

    Returns:
    - ``(None, None)`` when no same-actor goal exists;
    - ``(None, best_score)`` when candidates exist but none reaches threshold;
    - ``(matched_goal, best_score)`` when a duplicate is found.
    """
    candidate_actor = normalize_goal_name(candidate.actor.name)
    same_actor_goals = [
        goal
        for goal in existing_goals
        if normalize_goal_name(goal.actor.name) == candidate_actor
    ]

    if not same_actor_goals:
        return None, None

    similarity_matrix = _get_goal_evaluator().compute_similarity(
        [_high_level_goal_duplicate_text(candidate)],
        [_high_level_goal_duplicate_text(goal) for goal in same_actor_goals],
    )

    if not similarity_matrix.size:
        return None, None

    best_index = int(np.argmax(similarity_matrix[0]))
    best_score = float(similarity_matrix[0, best_index])

    if best_score >= threshold:
        return same_actor_goals[best_index], best_score

    return None, best_score


def is_semantic_duplicate_high_level_goal(
    candidate: HighLevelGoal,
    existing_goals: Sequence[HighLevelGoal],
    threshold: float = DEFAULT_HLG_DUPLICATE_SIMILARITY_THRESHOLD,
) -> bool:
    """Return whether ``candidate`` duplicates an existing same-actor HLG."""
    matched_goal, _ = find_semantic_duplicate_high_level_goal(
        candidate,
        existing_goals,
        threshold,
    )
    return matched_goal is not None


def _collection_semantically_equivalent(
    texts_a: list[str],
    texts_b: list[str],
    threshold: float,
) -> bool:
    """Compare two equal-sized collections with strict one-to-one matching."""
    if len(texts_a) != len(texts_b):
        return False
    if not texts_a:
        return True

    similarity_matrix = _get_goal_evaluator().compute_similarity(texts_a, texts_b)
    row_indices, column_indices = linear_sum_assignment(-similarity_matrix)
    matched_similarities = similarity_matrix[row_indices, column_indices]

    return bool(np.all(matched_similarities >= threshold))


def high_level_goals_semantically_equivalent(
    goals_a: Sequence[HighLevelGoal],
    goals_b: Sequence[HighLevelGoal],
    threshold: float = DEFAULT_STATE_SIMILARITY_THRESHOLD,
) -> bool:
    """Return whether two HLG collections are semantically equivalent."""
    return _collection_semantically_equivalent(
        [_high_level_goal_state_text(goal) for goal in goals_a],
        [_high_level_goal_state_text(goal) for goal in goals_b],
        threshold,
    )


def low_level_goals_semantically_equivalent(
    goals_a: Sequence[LowLevelGoal],
    goals_b: Sequence[LowLevelGoal],
    threshold: float = DEFAULT_STATE_SIMILARITY_THRESHOLD,
) -> bool:
    """Return whether two LLG collections are semantically equivalent."""
    return _collection_semantically_equivalent(
        [_low_level_goal_state_text(goal) for goal in goals_a],
        [_low_level_goal_state_text(goal) for goal in goals_b],
        threshold,
    )


def states_semantically_equivalent(
    high_level_goals_a: HighLevelGoals,
    high_level_goals_b: HighLevelGoals,
    low_level_goals_a: LowLevelGoals,
    low_level_goals_b: LowLevelGoals,
    high_level_threshold: float = DEFAULT_STATE_SIMILARITY_THRESHOLD,
    low_level_threshold: float = DEFAULT_STATE_SIMILARITY_THRESHOLD,
) -> bool:
    """Return whether both the HLG and LLG collections represent the same state."""
    high_level_equivalent = high_level_goals_semantically_equivalent(
        high_level_goals_a.goals,
        high_level_goals_b.goals,
        high_level_threshold,
    )
    low_level_equivalent = low_level_goals_semantically_equivalent(
        low_level_goals_a.low_level_goals,
        low_level_goals_b.low_level_goals,
        low_level_threshold,
    )

    return high_level_equivalent and low_level_equivalent
