"""
semantic_similarity.py

Shared semantic-similarity utilities for the bottom-up feedback loop:
- semantic duplicate detection among High-Level Goals (in addition to the
  existing lexical/name-based check);
- semantic repeated-state detection between successive cycle states (in
  addition to the existing exact SHA-256 state hash).

Both reuse the embeddings + cosine similarity principle already implemented
in ``src.evaluation.goal_evaluator.GoalEvaluator`` (mean-pooled transformer
embeddings, cosine similarity via normalized dot product, and prudent
one-to-one matching through ``scipy.optimize.linear_sum_assignment``),
instead of duplicating that logic here.

Threshold policy
-----------------
Every threshold exposed below is a configurable DEFAULT, not a definitive
value. The final thresholds used for High-Level-Goal duplicate detection and
for semantic repeated-state detection must be calibrated empirically on the
thesis' datasets / pilot experiments; they are intentionally kept as two
separate constants because there is no reason to assume the same cutoff is
appropriate for both questions ("are these two goals the same?" vs. "is the
whole cycle stuck oscillating between paraphrases of the same state?").
"""

from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.bottom_up.goal_reconstructor import normalize_goal_name
from src.data_model import HighLevelGoal, HighLevelGoals, LowLevelGoal, LowLevelGoals

# Cosine-similarity threshold above which two High-Level Goals (compared as
# actor + name + description) are considered semantic duplicates of one
# another. NOT a definitive value: to be calibrated empirically on the
# thesis' datasets/pilot experiments.
DEFAULT_HLG_DUPLICATE_SIMILARITY_THRESHOLD = 0.90

# Cosine-similarity threshold above which two matched goals (HLG or LLG) in a
# one-to-one matching between two cycle states are considered equivalent for
# semantic repeated-state detection. NOT a definitive value: to be calibrated
# empirically on the thesis' datasets/pilot experiments, independently from
# the HLG-duplicate threshold above.
DEFAULT_STATE_SIMILARITY_THRESHOLD = 0.92


@lru_cache(maxsize=1)
def _get_goal_evaluator():
    """
    Lazily build and reuse a single embedding-model instance.

    The import is deferred (rather than a module-level import) so that
    merely importing this module does not force loading the transformer
    model and running the nltk downloads performed by
    ``GoalEvaluator.__init__`` unless a semantic check is actually invoked.
    """
    from src.evaluation.goal_evaluator import GoalEvaluator

    return GoalEvaluator()


def _high_level_goal_text(goal: HighLevelGoal) -> str:
    """Textual representation of an HLG: actor + name + description."""
    return f"{goal.actor.name}. {goal.name}. {goal.description}"


def _low_level_goal_text(goal: LowLevelGoal) -> str:
    """
    Textual representation of an LLG, including its parent's context:
    parent actor + parent HLG + LLG name + LLG description.
    """
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
    Same actor-scoped, embeddings + cosine-similarity duplicate check as
    ``is_semantic_duplicate_high_level_goal``, but returns the matched goal
    and its similarity score instead of a bare bool, so a caller can build
    an informative message (which goal, how similar) instead of only
    knowing yes/no.

    Comparison is restricted to goals sharing the candidate's normalized
    actor name, for the same reason as ``is_semantic_duplicate_high_level_goal``.

    Returns ``(None, None)`` when no same-actor goal exists to compare
    against, and ``(None, best_score)`` when a same-actor pool exists but its
    best match stays below ``threshold``.
    """
    candidate_actor = normalize_goal_name(candidate.actor.name)
    same_actor_goals = [
        goal
        for goal in existing_goals
        if normalize_goal_name(goal.actor.name) == candidate_actor
    ]
    if not same_actor_goals:
        return None, None

    evaluator = _get_goal_evaluator()
    sim_matrix = evaluator.compute_similarity(
        [_high_level_goal_text(candidate)],
        [_high_level_goal_text(goal) for goal in same_actor_goals],
    )
    if not sim_matrix.size:
        return None, None

    best_index = int(np.argmax(sim_matrix[0]))
    best_score = float(sim_matrix[0, best_index])

    if best_score >= threshold:
        return same_actor_goals[best_index], best_score
    return None, best_score


def is_semantic_duplicate_high_level_goal(
    candidate: HighLevelGoal,
    existing_goals: Sequence[HighLevelGoal],
    threshold: float = DEFAULT_HLG_DUPLICATE_SIMILARITY_THRESHOLD,
) -> bool:
    """
    Returns True if ``candidate`` is a semantic duplicate of any goal in
    ``existing_goals``.

    Thin wrapper over ``find_semantic_duplicate_high_level_goal`` for callers
    that only need the yes/no answer.
    """
    matched, _ = find_semantic_duplicate_high_level_goal(
        candidate, existing_goals, threshold
    )
    return matched is not None


def _collection_semantically_equivalent(
    texts_a: list[str],
    texts_b: list[str],
    threshold: float,
) -> bool:
    """
    Prudent one-to-one (Hungarian) matching between two text collections.

    Equivalence is deliberately strict: it requires compatible cardinality
    (order is irrelevant, but the two collections must be the same size),
    a full one-to-one assignment, and EVERY matched pair - not merely the
    average similarity - to be at or above ``threshold``. A high average can
    hide one completely unrelated pair, which a prudent check must not miss.
    """
    if len(texts_a) != len(texts_b):
        return False
    if not texts_a:
        return True

    evaluator = _get_goal_evaluator()
    sim_matrix = evaluator.compute_similarity(texts_a, texts_b)

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    matched_similarities = sim_matrix[row_ind, col_ind]

    return bool(np.all(matched_similarities >= threshold))


def high_level_goals_semantically_equivalent(
    goals_a: Sequence[HighLevelGoal],
    goals_b: Sequence[HighLevelGoal],
    threshold: float = DEFAULT_STATE_SIMILARITY_THRESHOLD,
) -> bool:
    """Prudent semantic equivalence between two HLG collections."""
    return _collection_semantically_equivalent(
        [_high_level_goal_text(goal) for goal in goals_a],
        [_high_level_goal_text(goal) for goal in goals_b],
        threshold,
    )


def low_level_goals_semantically_equivalent(
    goals_a: Sequence[LowLevelGoal],
    goals_b: Sequence[LowLevelGoal],
    threshold: float = DEFAULT_STATE_SIMILARITY_THRESHOLD,
) -> bool:
    """Prudent semantic equivalence between two LLG collections."""
    return _collection_semantically_equivalent(
        [_low_level_goal_text(goal) for goal in goals_a],
        [_low_level_goal_text(goal) for goal in goals_b],
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
    """
    A cycle state is considered semantically equivalent to another only when
    BOTH its HLG collection and its LLG collection independently satisfy the
    prudent one-to-one matching criterion above.

    Only the goal collections themselves are compared. Evaluator rationale,
    generation descriptions, and any other free text are never part of this
    comparison.
    """
    return high_level_goals_semantically_equivalent(
        high_level_goals_a.goals,
        high_level_goals_b.goals,
        high_level_threshold,
    ) and low_level_goals_semantically_equivalent(
        low_level_goals_a.low_level_goals,
        low_level_goals_b.low_level_goals,
        low_level_threshold,
    )
