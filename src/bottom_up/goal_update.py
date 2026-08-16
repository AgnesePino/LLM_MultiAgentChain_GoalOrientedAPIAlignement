"""
goal_update.py

Helpers that apply Global-Goal-Evaluator decisions to the current HLG/LLG
state: HLG deduplication, collecting and executing HLG-generation requests,
applying additions/replacements deterministically, selecting branches for
low-level regeneration, and merging selectively regenerated low-level goals.

The module also owns evaluator-backed duplicate resolution after HLG generation: semantic similarity identifies duplicate clusters, the Global Goal Evaluator selects the best project-grounded GORE HLG, and this layer deterministically removes the losing generated/existing goals.
"""

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from src.data_model import HighLevelGoal, HighLevelGoals, LowLevelGoal, LowLevelGoals
from src.bottom_up.goal_reconstructor import normalize_goal_name
from src.bottom_up.models import (
    GlobalGoalEvaluationDecision,
    GlobalGoalEvaluationResult,
    HighLevelGoalGenerationAction,
    HighLevelGoalGenerationRequest,
    HighLevelGoalGenerationSource,
    HighLevelGoalDuplicateCandidateSource,
    HighLevelGoalDuplicateResolution,
)
from src.bottom_up.semantic_similarity import (
    find_semantic_duplicate_high_level_goal,
    is_semantic_duplicate_high_level_goal,
)
from src.llm_clients import MAX_SEMANTIC_RETRIES

# The injected callback receives one validated request and must internally call
# the original top-down HLG generator with request.generator_input and
# feedback=None. The original generator may return one or more HLGs.
HighLevelGoalGenerator = Callable[
    [HighLevelGoalGenerationRequest],
    HighLevelGoals,
]

# The injected callback receives only the HLGs whose LLGs must be generated.
LowLevelGoalRegenerator = Callable[[HighLevelGoals], LowLevelGoals]

# The injected evaluator callback receives one semantic-duplicate cluster.
# Besides the candidates themselves, the update layer supplies the complete
# project description and traceability metadata so the evaluator can select
# the candidate that is best supported by the project documentation and best
# satisfies the GORE High-Level Goal abstraction criteria.  It may select only
# one of the supplied candidates; it never creates, merges, or rewrites goals.
class HighLevelGoalDuplicateSelector(Protocol):
    def __call__(
        self,
        request: HighLevelGoalGenerationRequest,
        candidates: dict[str, HighLevelGoal],
        *,
        project_description: str,
        candidate_sources: dict[
            str, HighLevelGoalDuplicateCandidateSource
        ],
        candidate_branch_ids: dict[str, str | None],
    ) -> tuple[str, str]: ...


@dataclass(frozen=True)
class _DuplicateCandidateRecord:
    """Internal traceability for one candidate in a duplicate cluster."""

    goal: HighLevelGoal
    source: HighLevelGoalDuplicateCandidateSource
    branch_id: str | None
    owner_request_id: str | None
    is_current_request_candidate: bool = False


def _deduplicate_goals(goals: list[HighLevelGoal]) -> HighLevelGoals:
    """
    Deduplicate high-level goals.

    Lexical normalized-name equality is checked first, as it is cheap and
    deterministic. A goal that survives the lexical check is then compared,
    within the same actor, against every goal already kept, using the
    embeddings + cosine-similarity semantic check from
    ``semantic_similarity``: two differently worded goals expressing the
    same intention must not both survive deduplication.
    """
    result: list[HighLevelGoal] = []
    seen: set[str] = set()

    for goal in goals:
        key = normalize_goal_name(goal.name)
        if key in seen:
            continue
        if is_semantic_duplicate_high_level_goal(goal, result):
            continue
        seen.add(key)
        result.append(goal)

    return HighLevelGoals(goals=result)


def _collect_high_level_generation_requests(
    evaluations: dict[str, GlobalGoalEvaluationResult],
) -> list[HighLevelGoalGenerationRequest]:
    """
    Collect the validated HLG-generation requests emitted by branch evaluation.

    The function also verifies that the orchestration metadata in each request
    is consistent with the evaluator decision and dictionary branch key.
    """
    requests: list[HighLevelGoalGenerationRequest] = []
    seen_request_ids: set[str] = set()

    for branch_id, evaluation in evaluations.items():
        if not evaluation.requires_high_level_regeneration:
            continue

        request = evaluation.generation_request
        if request is None:
            raise ValueError(
                f"Branch '{branch_id}' requires HLG generation but contains "
                "no generation_request."
            )

        if request.origin_branch_id != branch_id:
            raise ValueError(
                f"Request '{request.request_id}' has origin_branch_id "
                f"'{request.origin_branch_id}', expected '{branch_id}'."
            )

        if (
            evaluation.decision
            == GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
        ):
            if (
                request.action
                != HighLevelGoalGenerationAction.REPLACE_EXISTING_HIGH_LEVEL_GOAL
            ):
                raise ValueError(
                    f"Branch '{branch_id}' requires a replacement request, "
                    f"but action is '{request.action.value}'."
                )
            if request.target_branch_id != branch_id:
                raise ValueError(
                    f"Request '{request.request_id}' targets branch "
                    f"'{request.target_branch_id}', expected '{branch_id}'."
                )

        elif (
            evaluation.decision
            == GlobalGoalEvaluationDecision.ADD_NEW_HIGH_LEVEL_GOAL
        ):
            if (
                request.action
                != HighLevelGoalGenerationAction.ADD_NEW_HIGH_LEVEL_GOAL
            ):
                raise ValueError(
                    f"Branch '{branch_id}' requires an addition request, "
                    f"but action is '{request.action.value}'."
                )
            if request.target_branch_id is not None:
                raise ValueError(
                    f"Addition request '{request.request_id}' must not target "
                    "an existing branch."
                )

        else:
            raise ValueError(
                f"Branch '{branch_id}' unexpectedly requires HLG generation "
                f"for decision '{evaluation.decision.value}'."
            )

        if request.request_id in seen_request_ids:
            raise ValueError(
                f"Duplicate HLG generation request id: '{request.request_id}'."
            )

        seen_request_ids.add(request.request_id)
        requests.append(request)

    return requests


def _generate_requested_high_level_goals(
    generate_high_level_goals: HighLevelGoalGenerator,
    requests: list[HighLevelGoalGenerationRequest],
    existing_high_level_goals: dict[str, HighLevelGoal],
    select_best_duplicate_high_level_goal: HighLevelGoalDuplicateSelector,
    project_description: str | None = None,
) -> tuple[
    dict[str, list[HighLevelGoal]],
    list[HighLevelGoalDuplicateResolution],
    str | None,
]:
    """
    Execute HLG-generation requests and resolve every duplicate cluster that
    intersects the newly generated candidates.

    Duplicate detection and duplicate quality selection are deliberately
    separate operations:
    - semantic similarity / normalized-name equality identify a cluster;
    - the Global Goal Evaluator chooses the single best representative using
      the complete project description and GORE HLG criteria;
    - deterministic Python logic removes every losing candidate.

    A cluster may contain:
    - multiple HLGs returned by the current generator response;
    - HLGs already present in the current cycle state;
    - HLGs accepted from earlier requests in the same generation batch.

    Therefore a semantic duplicate is no longer a reason to retry the same
    top-down request or abort the whole batch.  The request may legitimately
    end with zero newly generated HLGs when an already-existing/prior candidate
    wins the duplicate selection.  Processing then continues with the next
    request.

    For REPLACE_EXISTING_HIGH_LEVEL_GOAL, the explicit target branch is not
    included in duplicate competition for that same request: a replacement is
    expected to preserve the underlying intention of the branch it replaces.
    It is still compared against every *other* HLG in the state.
    """
    generated_by_request: dict[str, list[HighLevelGoal]] = {}
    duplicate_resolutions: list[HighLevelGoalDuplicateResolution] = []
    discarded_existing_branch_ids: set[str] = set()

    for request in requests:
        validated_goals: list[HighLevelGoal] | None = None
        validation_error: str | None = None

        # Keep the existing bounded retry for malformed/invalid generator
        # output.  Semantic duplicates are intentionally NOT validation errors
        # anymore, so they never cause the same request to be regenerated just
        # to obtain cosmetically different wording.
        for _ in range(MAX_SEMANTIC_RETRIES + 1):
            try:
                generated = generate_high_level_goals(request)
            except Exception as exc:
                return {}, [], (
                    f"{type(exc).__name__}: HLG request "
                    f"'{request.request_id}' failed: {exc}"
                )

            validated_goals, validation_error = (
                _validate_generated_high_level_goals(
                    request=request,
                    generated=generated,
                )
            )
            if validation_error is None:
                break

        if validation_error is not None:
            return {}, [], validation_error

        assert validated_goals is not None
        generated_by_request[request.request_id] = list(validated_goals)

        selector_project_description = (
            project_description
            if project_description is not None
            else request.generator_input.project_description
        )
        if not selector_project_description.strip():
            return {}, [], (
                "ValueError: duplicate resolution requires a non-empty "
                "complete project description."
            )

        try:
            request_resolutions, newly_discarded_existing = (
                _resolve_high_level_goal_duplicates_for_request(
                    request=request,
                    existing_high_level_goals=existing_high_level_goals,
                    generated_by_request=generated_by_request,
                    discarded_existing_branch_ids=(
                        discarded_existing_branch_ids
                    ),
                    project_description=selector_project_description,
                    select_best_duplicate_high_level_goal=(
                        select_best_duplicate_high_level_goal
                    ),
                )
            )
        except Exception as exc:
            return {}, [], (
                f"{type(exc).__name__}: duplicate resolution for HLG request "
                f"'{request.request_id}' failed: {exc}"
            )

        duplicate_resolutions.extend(request_resolutions)
        discarded_existing_branch_ids.update(newly_discarded_existing)

    return generated_by_request, duplicate_resolutions, None


def _validate_generated_high_level_goals(
    request: HighLevelGoalGenerationRequest,
    generated: HighLevelGoals,
) -> tuple[list[HighLevelGoal] | None, str | None]:
    """
    Validate one raw top-down generator response before duplicate resolution.

    This stage checks only conditions that make a candidate intrinsically
    unusable: wrong return type, empty response, empty name/description, or an
    actor that was not supplied to the generator.  Semantic overlap is not an
    error here.  It is handled afterwards by evaluator-backed duplicate
    selection so the best HLG survives instead of repeatedly calling the same
    generator request.
    """
    if not isinstance(generated, HighLevelGoals):
        return None, (
            "TypeError: generate_high_level_goals must return a "
            "HighLevelGoals instance."
        )

    if not generated.goals:
        return None, (
            f"ValueError: request '{request.request_id}' returned no HLGs."
        )

    allowed_actor_names = {
        normalize_goal_name(actor.name)
        for actor in request.generator_input.actors.actors
    }
    validated_goals: list[HighLevelGoal] = []

    for generated_goal in generated.goals:
        if (
            not generated_goal.name.strip()
            or not generated_goal.description.strip()
        ):
            return None, (
                f"ValueError: request '{request.request_id}' returned an "
                "HLG with an empty name or description."
            )

        returned_actor_name = normalize_goal_name(generated_goal.actor.name)
        if returned_actor_name not in allowed_actor_names:
            return None, (
                f"ValueError: request '{request.request_id}' returned "
                f"actor '{generated_goal.actor.name}', which was not "
                "supplied to the top-down generator."
            )

        validated_goals.append(generated_goal)

    return validated_goals, None


def _goals_are_duplicates(
    left_goal: HighLevelGoal,
    right_goal: HighLevelGoal,
) -> bool:
    """Return whether two HLGs belong to the same duplicate relation."""
    if normalize_goal_name(left_goal.name) == normalize_goal_name(
        right_goal.name
    ):
        return True

    matched, _ = find_semantic_duplicate_high_level_goal(
        left_goal,
        [right_goal],
    )
    return matched is not None


def _find_duplicate_candidate_clusters(
    records: Sequence[_DuplicateCandidateRecord],
) -> list[list[int]]:
    """Return connected components of the duplicate relation.

    Connected components are intentionally used rather than greedy filtering:
    if A≈B and B≈C, the selector sees A/B/C together and chooses one global
    representative even when A and C are just below the pairwise threshold.
    Only components containing a candidate generated by the *current* request
    are returned; unrelated duplicates already present elsewhere in the state
    are not modified opportunistically.
    """
    count = len(records)
    if count < 2:
        return []

    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(count):
        for right in range(left + 1, count):
            if _goals_are_duplicates(records[left].goal, records[right].goal):
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    clusters: list[list[int]] = []
    for _, indices in sorted(groups.items(), key=lambda item: min(item[1])):
        if len(indices) < 2:
            continue
        if not any(records[index].is_current_request_candidate for index in indices):
            continue
        clusters.append(indices)
    return clusters


def _remove_generated_record_from_owner(
    record: _DuplicateCandidateRecord,
    generated_by_request: dict[str, list[HighLevelGoal]],
) -> None:
    """Remove exactly one losing generated candidate from its owner request."""
    owner_request_id = record.owner_request_id
    if owner_request_id is None:
        raise ValueError("Generated duplicate candidate has no owner request id.")

    owner_goals = generated_by_request.get(owner_request_id)
    if owner_goals is None:
        raise ValueError(
            f"Generated duplicate candidate references unknown owner request "
            f"'{owner_request_id}'."
        )

    for index, goal in enumerate(owner_goals):
        if goal is record.goal:
            del owner_goals[index]
            return

    # Defensive fallback for callers that may have copied Pydantic models.
    for index, goal in enumerate(owner_goals):
        if goal == record.goal:
            del owner_goals[index]
            return

    raise ValueError(
        f"Could not remove losing generated HLG '{record.goal.name}' from "
        f"owner request '{owner_request_id}'."
    )


def _build_duplicate_candidate_records(
    request: HighLevelGoalGenerationRequest,
    existing_high_level_goals: dict[str, HighLevelGoal],
    generated_by_request: dict[str, list[HighLevelGoal]],
    discarded_existing_branch_ids: set[str],
) -> list[_DuplicateCandidateRecord]:
    """Build the active candidate universe for one request's resolution."""
    records: list[_DuplicateCandidateRecord] = []

    # Current generated candidates come first so candidate ids remain easy to
    # interpret in traces.  Earlier generated winners follow, then existing
    # HLGs in branch-map order.
    for goal in generated_by_request.get(request.request_id, []):
        records.append(
            _DuplicateCandidateRecord(
                goal=goal,
                source=HighLevelGoalDuplicateCandidateSource.GENERATED,
                branch_id=None,
                owner_request_id=request.request_id,
                is_current_request_candidate=True,
            )
        )

    for owner_request_id, goals in generated_by_request.items():
        if owner_request_id == request.request_id:
            continue
        for goal in goals:
            records.append(
                _DuplicateCandidateRecord(
                    goal=goal,
                    source=HighLevelGoalDuplicateCandidateSource.GENERATED,
                    branch_id=None,
                    owner_request_id=owner_request_id,
                )
            )

    for branch_id, goal in existing_high_level_goals.items():
        if branch_id in discarded_existing_branch_ids:
            continue
        if (
            request.action
            == HighLevelGoalGenerationAction.REPLACE_EXISTING_HIGH_LEVEL_GOAL
            and branch_id == request.target_branch_id
        ):
            continue
        records.append(
            _DuplicateCandidateRecord(
                goal=goal,
                source=HighLevelGoalDuplicateCandidateSource.EXISTING,
                branch_id=branch_id,
                owner_request_id=None,
            )
        )

    return records


def _resolve_high_level_goal_duplicates_for_request(
    request: HighLevelGoalGenerationRequest,
    existing_high_level_goals: dict[str, HighLevelGoal],
    generated_by_request: dict[str, list[HighLevelGoal]],
    discarded_existing_branch_ids: set[str],
    project_description: str,
    select_best_duplicate_high_level_goal: HighLevelGoalDuplicateSelector,
) -> tuple[list[HighLevelGoalDuplicateResolution], set[str]]:
    """
    Resolve all duplicate components touched by the current request.

    The selector receives GENERATED and EXISTING candidates together.  Python
    then removes every non-selected generated candidate immediately and marks
    every non-selected existing branch for deterministic deletion when the HLG
    state is applied.
    """
    records = _build_duplicate_candidate_records(
        request=request,
        existing_high_level_goals=existing_high_level_goals,
        generated_by_request=generated_by_request,
        discarded_existing_branch_ids=discarded_existing_branch_ids,
    )
    clusters = _find_duplicate_candidate_clusters(records)
    if not clusters:
        return [], set()

    resolutions: list[HighLevelGoalDuplicateResolution] = []
    newly_discarded_existing: set[str] = set()

    # Components are disjoint.  We can therefore resolve against the snapshot
    # above and apply all losing-candidate removals afterwards without changing
    # membership of another component.
    generated_records_to_remove: list[_DuplicateCandidateRecord] = []

    for cluster_indices in clusters:
        cluster_records = [records[index] for index in cluster_indices]
        candidate_ids = [
            f"candidate_{index + 1:03d}"
            for index in range(len(cluster_records))
        ]
        candidates = {
            candidate_id: record.goal
            for candidate_id, record in zip(
                candidate_ids, cluster_records, strict=True
            )
        }
        candidate_sources = {
            candidate_id: record.source
            for candidate_id, record in zip(
                candidate_ids, cluster_records, strict=True
            )
        }
        candidate_branch_ids = {
            candidate_id: record.branch_id
            for candidate_id, record in zip(
                candidate_ids, cluster_records, strict=True
            )
        }

        selected_candidate_id, rationale = (
            select_best_duplicate_high_level_goal(
                request,
                candidates,
                project_description=project_description,
                candidate_sources=candidate_sources,
                candidate_branch_ids=candidate_branch_ids,
            )
        )
        if selected_candidate_id not in candidates:
            raise ValueError(
                "Duplicate selector returned an unknown candidate id "
                f"'{selected_candidate_id}' for request "
                f"'{request.request_id}'."
            )

        selected_position = candidate_ids.index(selected_candidate_id)
        selected_record = cluster_records[selected_position]
        losing_records = [
            record
            for index, record in enumerate(cluster_records)
            if index != selected_position
        ]

        for losing_record in losing_records:
            if (
                losing_record.source
                == HighLevelGoalDuplicateCandidateSource.EXISTING
            ):
                if losing_record.branch_id is None:
                    raise ValueError(
                        "Existing duplicate candidate is missing its branch id."
                    )
                newly_discarded_existing.add(losing_record.branch_id)
            else:
                generated_records_to_remove.append(losing_record)

        resolutions.append(
            HighLevelGoalDuplicateResolution(
                request_id=request.request_id,
                candidate_ids=candidate_ids,
                candidate_sources=candidate_sources,
                candidate_branch_ids=candidate_branch_ids,
                selected_candidate_id=selected_candidate_id,
                selected_goal=selected_record.goal,
                discarded_goals=[record.goal for record in losing_records],
                rationale=rationale,
            )
        )

    # A generated goal can belong to only one connected component, hence it is
    # removed at most once.  This also updates goals accepted by earlier
    # requests when a later request produces a better duplicate.
    seen_generated_objects: set[int] = set()
    for record in generated_records_to_remove:
        object_id = id(record.goal)
        if object_id in seen_generated_objects:
            continue
        seen_generated_objects.add(object_id)
        _remove_generated_record_from_owner(
            record=record,
            generated_by_request=generated_by_request,
        )

    return resolutions, newly_discarded_existing


def _find_generated_duplicate_clusters(
    generated_goals: Sequence[HighLevelGoal],
) -> list[list[int]]:
    """Backward-compatible helper for duplicate clusters in one response."""
    records = [
        _DuplicateCandidateRecord(
            goal=goal,
            source=HighLevelGoalDuplicateCandidateSource.GENERATED,
            branch_id=None,
            owner_request_id="compatibility",
            is_current_request_candidate=True,
        )
        for goal in generated_goals
    ]
    return _find_duplicate_candidate_clusters(records)


def _resolve_generated_high_level_goal_duplicates(
    request: HighLevelGoalGenerationRequest,
    generated_goals: list[HighLevelGoal],
    select_best_duplicate_high_level_goal: HighLevelGoalDuplicateSelector,
    project_description: str | None = None,
) -> tuple[list[HighLevelGoal], list[HighLevelGoalDuplicateResolution]]:
    """
    Backward-compatible intra-response duplicate resolver.

    New orchestration should use `_generate_requested_high_level_goals`, which
    can resolve mixed GENERATED/EXISTING clusters.  This wrapper preserves the
    previous helper for any external experiments that import it directly.
    """
    if not generated_goals:
        return [], []

    generated_by_request = {request.request_id: list(generated_goals)}
    resolutions, _ = _resolve_high_level_goal_duplicates_for_request(
        request=request,
        existing_high_level_goals={},
        generated_by_request=generated_by_request,
        discarded_existing_branch_ids=set(),
        project_description=(
            project_description
            if project_description is not None
            else request.generator_input.project_description
        ),
        select_best_duplicate_high_level_goal=(
            select_best_duplicate_high_level_goal
        ),
    )
    return generated_by_request[request.request_id], resolutions


def _flatten_generated_high_level_goals(
    generated_by_request: dict[str, list[HighLevelGoal]],
) -> list[HighLevelGoal]:
    """Flatten generated HLG lists while preserving request/output order."""
    return [
        goal
        for goals in generated_by_request.values()
        for goal in goals
    ]


def _discarded_existing_branch_ids(
    duplicate_resolutions: Sequence[HighLevelGoalDuplicateResolution],
) -> set[str]:
    return {
        branch_id
        for resolution in duplicate_resolutions
        for branch_id in resolution.discarded_existing_branch_ids
    }


def _discarded_existing_goal_names(
    duplicate_resolutions: Sequence[HighLevelGoalDuplicateResolution],
) -> set[str]:
    """Return normalized names of existing HLGs removed by duplicate choice."""
    names: set[str] = set()
    for resolution in duplicate_resolutions:
        losing_ids = [
            candidate_id
            for candidate_id in resolution.candidate_ids
            if candidate_id != resolution.selected_candidate_id
        ]
        for candidate_id, goal in zip(
            losing_ids, resolution.discarded_goals, strict=True
        ):
            if (
                resolution.candidate_sources[candidate_id]
                == HighLevelGoalDuplicateCandidateSource.EXISTING
            ):
                names.add(normalize_goal_name(goal.name))
    return names


def _apply_branch_high_level_generation(
    existing_high_level_goals: dict[str, HighLevelGoal],
    requests: list[HighLevelGoalGenerationRequest],
    generated_by_request: dict[str, list[HighLevelGoal]],
    duplicate_resolutions: Sequence[HighLevelGoalDuplicateResolution] = (),
) -> tuple[HighLevelGoals, list[HighLevelGoal]]:
    """
    Apply branch ADD/REPLACE requests after evaluator-backed deduplication.

    Existing HLGs that lost a duplicate-selection cluster are removed.  A
    request is also allowed to have no surviving newly generated HLG when an
    existing or earlier-generated candidate won its duplicate cluster.
    """
    replacements: dict[str, list[HighLevelGoal]] = {}
    additions: list[HighLevelGoal] = []
    discarded_branch_ids = _discarded_existing_branch_ids(
        duplicate_resolutions
    )

    for request in requests:
        if request.request_id not in generated_by_request:
            raise ValueError(
                f"No generation result exists for request "
                f"'{request.request_id}'."
            )
        # An empty list is valid here: this request did produce HLGs, but all
        # of them were later removed because another GENERATED or EXISTING HLG
        # won an evaluator-backed duplicate cluster.
        generated_goals = generated_by_request[request.request_id]

        if (
            request.action
            == HighLevelGoalGenerationAction.REPLACE_EXISTING_HIGH_LEVEL_GOAL
        ):
            target_branch_id = request.target_branch_id
            if target_branch_id is None:
                raise ValueError(
                    f"Request '{request.request_id}' has no target branch."
                )
            if target_branch_id not in existing_high_level_goals:
                raise ValueError(
                    f"Request '{request.request_id}' references unknown branch "
                    f"'{target_branch_id}'."
                )
            if target_branch_id in replacements:
                raise ValueError(
                    "Multiple HLG replacements target branch "
                    f"'{target_branch_id}'."
                )
            # An empty list means that the intended replacement was absorbed by
            # another candidate selected from a duplicate cluster: remove the
            # old target but do not create a second equivalent branch.
            replacements[target_branch_id] = list(generated_goals)

        elif (
            request.action
            == HighLevelGoalGenerationAction.ADD_NEW_HIGH_LEVEL_GOAL
        ):
            additions.extend(generated_goals)

        else:
            raise ValueError(
                f"Unsupported HLG generation action '{request.action.value}'."
            )

    updated: list[HighLevelGoal] = []
    seen_names: set[str] = set()

    def append_unique(goal: HighLevelGoal) -> None:
        normalized_name = normalize_goal_name(goal.name)
        if normalized_name in seen_names:
            raise ValueError(
                f"HLG generation produced duplicate goal name '{goal.name}' "
                "after evaluator-backed duplicate resolution."
            )
        if is_semantic_duplicate_high_level_goal(goal, updated):
            raise ValueError(
                "HLG generation still contains a semantic duplicate after "
                f"evaluator-backed resolution: '{goal.name}'."
            )
        seen_names.add(normalized_name)
        updated.append(goal)

    for branch_id, original in existing_high_level_goals.items():
        replacement_goals = replacements.get(branch_id)
        if replacement_goals is not None:
            for replacement_goal in replacement_goals:
                append_unique(replacement_goal)
            continue

        if branch_id in discarded_branch_ids:
            continue

        append_unique(original)

    for new_goal in additions:
        append_unique(new_goal)

    return HighLevelGoals(goals=updated), additions


def _append_coverage_generated_goals(
    current_high_level_goals: HighLevelGoals,
    requests: list[HighLevelGoalGenerationRequest],
    generated_by_request: dict[str, list[HighLevelGoal]],
    duplicate_resolutions: Sequence[HighLevelGoalDuplicateResolution] = (),
) -> tuple[HighLevelGoals, list[HighLevelGoal]]:
    """
    Apply documentation-coverage generation after duplicate resolution.

    If an existing HLG wins, the generated duplicate is simply absent from the
    surviving generated list.  If a generated HLG wins against an existing
    duplicate, the losing existing branch is removed and the generated winner
    is appended.  Multiple losing existing duplicates are all removed.
    """
    discarded_branch_ids = _discarded_existing_branch_ids(
        duplicate_resolutions
    )

    # branch ids are assigned by goal_reconstructor.assign_branch_ids exactly
    # in list order (branch_001, branch_002, ...), so the resolution metadata
    # can be applied deterministically without invoking the reconstructor here.
    current_goals = [
        goal
        for index, goal in enumerate(current_high_level_goals.goals, start=1)
        if f"branch_{index:03d}" not in discarded_branch_ids
    ]
    seen_names = {
        normalize_goal_name(goal.name)
        for goal in current_goals
    }
    added: list[HighLevelGoal] = []

    for request in requests:
        if (
            request.source
            != HighLevelGoalGenerationSource.DOCUMENTATION_COVERAGE
        ):
            raise ValueError(
                f"Coverage request '{request.request_id}' has invalid source "
                f"'{request.source.value}'."
            )

        if (
            request.action
            != HighLevelGoalGenerationAction.ADD_NEW_HIGH_LEVEL_GOAL
        ):
            raise ValueError(
                "Documentation coverage may only add new HLGs."
            )

        if request.request_id not in generated_by_request:
            raise ValueError(
                f"No generation result exists for coverage request "
                f"'{request.request_id}'."
            )
        generated_goals = generated_by_request[request.request_id]

        for generated_goal in generated_goals:
            normalized_name = normalize_goal_name(generated_goal.name)
            if normalized_name in seen_names:
                raise ValueError(
                    f"Coverage-generated HLG '{generated_goal.name}' still "
                    "duplicates an existing goal after duplicate resolution."
                )

            matched_existing_goal, similarity = (
                find_semantic_duplicate_high_level_goal(
                    generated_goal, current_goals
                )
            )
            if matched_existing_goal is not None:
                raise ValueError(
                    f"Coverage-generated HLG '{generated_goal.name}' still "
                    f"semantically duplicates existing HLG "
                    f"'{matched_existing_goal.name}' after evaluator-backed "
                    f"resolution (similarity={similarity:.4f})."
                )

            seen_names.add(normalized_name)
            current_goals.append(generated_goal)
            added.append(generated_goal)

    return HighLevelGoals(goals=current_goals), added


def _collect_branch_regeneration_targets(
    branch_map: dict[str, HighLevelGoal],
    evaluations: dict[str, GlobalGoalEvaluationResult],
    generated_by_request: dict[str, list[HighLevelGoal]],
    duplicate_resolutions: Sequence[HighLevelGoalDuplicateResolution] = (),
) -> tuple[HighLevelGoals, set[str]]:
    """
    Return HLGs whose LLGs must be regenerated and parent names to remove.

    Duplicate selection can remove an existing branch even when that branch's
    own evaluator decision was CONFIRM_BRANCH.  Such losing branches therefore
    contribute their parent name to `replaced_parent_names`, while no new LLGs
    are generated for the removed HLG itself.  Surviving generated winners are
    decomposed normally.
    """
    if set(branch_map) != set(evaluations):
        missing = set(branch_map) - set(evaluations)
        unexpected = set(evaluations) - set(branch_map)
        raise ValueError(
            "Cannot collect regeneration targets from an incomplete "
            f"evaluation set. Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}."
        )

    targets: list[HighLevelGoal] = []
    replaced_parent_names = _discarded_existing_goal_names(
        duplicate_resolutions
    )
    discarded_branch_ids = _discarded_existing_branch_ids(
        duplicate_resolutions
    )
    for branch_id, original in branch_map.items():
        evaluation = evaluations[branch_id]
        original_survives = branch_id not in discarded_branch_ids

        if evaluation.decision == GlobalGoalEvaluationDecision.CONFIRM_BRANCH:
            # A confirmed branch normally stays untouched.  If it lost a
            # duplicate-selection cluster, its LLGs are removed through
            # replaced_parent_names and no decomposition is requested for it.
            continue

        replaced_parent_names.add(normalize_goal_name(original.name))

        if (
            evaluation.decision
            == GlobalGoalEvaluationDecision.REWRITE_ORIGINAL_HIGH_LEVEL_GOAL
        ):
            request = evaluation.generation_request
            if request is None:
                raise ValueError(
                    f"Branch '{branch_id}' has no generation request."
                )

            if request.request_id not in generated_by_request:
                raise ValueError(
                    f"Branch '{branch_id}' has no generation result for "
                    f"request '{request.request_id}'."
                )
            replacements = generated_by_request[request.request_id]
            targets.extend(replacements)

        elif (
            evaluation.decision
            == GlobalGoalEvaluationDecision.ADD_NEW_HIGH_LEVEL_GOAL
        ):
            request = evaluation.generation_request
            if request is None:
                raise ValueError(
                    f"Branch '{branch_id}' has no generation request."
                )

            if request.request_id not in generated_by_request:
                raise ValueError(
                    f"Branch '{branch_id}' has no generation result for "
                    f"request '{request.request_id}'."
                )
            new_goals = generated_by_request[request.request_id]

            # The ADD decision still means the original branch's decomposition
            # requires revision.  Do not regenerate it only if the branch was
            # itself removed because it lost another duplicate cluster.
            if original_survives:
                targets.append(original)
            targets.extend(new_goals)

        elif evaluation.decision in {
            GlobalGoalEvaluationDecision.REGENERATE_LOW_LEVEL_GOALS,
            GlobalGoalEvaluationDecision.MATCHES_OTHER_HIGH_LEVEL_GOAL,
        }:
            if original_survives:
                targets.append(original)

        else:
            raise ValueError(
                f"Branch '{branch_id}' has unsupported decision "
                f"'{evaluation.decision.value}'."
            )

    return _deduplicate_goals(targets), replaced_parent_names


def _merge_selectively_regenerated_low_level_goals(
    current_low_level_goals: LowLevelGoals,
    regenerated_low_level_goals: LowLevelGoals,
    replaced_parent_names: set[str],
) -> LowLevelGoals:
    """Preserve confirmed branches and replace only requested branches."""
    preserved: list[LowLevelGoal] = [
        goal
        for goal in current_low_level_goals.low_level_goals
        if normalize_goal_name(goal.high_level_associated.name)
        not in replaced_parent_names
    ]

    return LowLevelGoals(
        low_level_goals=[
            *preserved,
            *regenerated_low_level_goals.low_level_goals,
        ]
    )


def _regenerate_selected_branches(
    regenerate_low_level_goals: LowLevelGoalRegenerator,
    targets: HighLevelGoals,
    current_low_level_goals: LowLevelGoals,
    replaced_parent_names: set[str],
) -> tuple[LowLevelGoals | None, str | None]:
    # Duplicate resolution can legitimately absorb every generated candidate
    # into an already-existing HLG.  In that case there may be no HLG requiring
    # a fresh decomposition.  Still apply any deterministic removal of LLGs
    # belonging to existing HLGs that lost a duplicate cluster, and otherwise
    # preserve the current LLG state unchanged.
    if not targets.goals:
        if replaced_parent_names:
            preserved = [
                goal
                for goal in current_low_level_goals.low_level_goals
                if normalize_goal_name(goal.high_level_associated.name)
                not in replaced_parent_names
            ]
            return LowLevelGoals(low_level_goals=preserved), None
        return current_low_level_goals.model_copy(deep=True), None

    try:
        regenerated = regenerate_low_level_goals(targets)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if not isinstance(regenerated, LowLevelGoals):
        return None, (
            "TypeError: regenerate_low_level_goals must return "
            "a LowLevelGoals instance."
        )

    expected_parent_names = {
        normalize_goal_name(goal.name)
        for goal in targets.goals
    }
    returned_parent_names = {
        normalize_goal_name(goal.high_level_associated.name)
        for goal in regenerated.low_level_goals
    }

    missing = expected_parent_names - returned_parent_names
    unexpected = returned_parent_names - expected_parent_names
    if missing or unexpected:
        return None, (
            "ValueError: selective regeneration returned an inconsistent set "
            f"of parent goals. Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}."
        )

    merged = _merge_selectively_regenerated_low_level_goals(
        current_low_level_goals=current_low_level_goals,
        regenerated_low_level_goals=regenerated,
        replaced_parent_names=replaced_parent_names,
    )
    return merged, None
