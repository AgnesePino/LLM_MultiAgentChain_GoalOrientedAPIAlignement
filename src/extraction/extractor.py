from src.examples.shot_learning import (
    ShotPromptingMode,
    example1_actors,
    example2_actors,
    example1_hl,
    example2_hl,
    example1_ll,
    example2_ll,
)
from src.data_model import (
    DocumentDescription,
    Actors,
    LowLevelGoals,
    HighLevelGoals,
    HighLevelGoalGenerationRequest,
    LowLevelGoalRegenerationRequest,
)
from src.utils import get_markdown
from src.llm_clients import generate_response


def generate_description(documentation_link=None):
    """Summarize a project's README into a stakeholder-style description."""
    if documentation_link is None:
        raise Exception("No documentation link provided")

    sys_prompt = (
        "You are a technical writing assistant specialized in summarizing "
        "software documentation. Your goal is to extract a clear, well-written, "
        "and accurate description of a project from its README file. The "
        "description should be natural and informative, without unnecessary "
        "details or implementation specifics. Avoid marketing language, vague "
        "claims, or filler content. Talk as a stakeholder describing the system "
        "they want to be implemented during requirements elicitation."
    )

    prompt = (
        f"Here is the README file of a software project:\n\n"
        f"{get_markdown(link=documentation_link)}\n\n"
        "Based on this README, write a well-structured description of the "
        "project. Explain its purpose, the problem it addresses (if mentioned), "
        "and its main functionalities. Do not include software implementation "
        "details."
    )

    return generate_response(prompt, sys_prompt, DocumentDescription)


def generate_actors(
    project_description,
    feedback=None,
    mode=ShotPromptingMode.ZERO_SHOT,
):
    """Original top-down actor extractor, with an optional critique retry."""
    sys_prompt = (
        "You are a helpful assistant expert in software engineering tasks, "
        "specialized in extracting end-user roles from a high-level description "
        "of a software project.\n"
        "Your task is to extract the actors (roles of end users of the system) "
        "from the given description.\n"
        "If actors are not explicitly mentioned, infer them based on typical "
        "users of similar software systems. Each extracted actor name should be "
        "accompanied by a very short description.\n"
    )

    if feedback is not None:
        print("Feedback provided!")
        sys_prompt += f"""

        The task given to you was already attempted but its output was flawed.
        You are provided with a critique on the previous attempt. Modify only
        what is mentioned in the critique.

        **Previous attempt:**
        {feedback.previous_output}

        **Critique:**
        {feedback.critique}
        """
    else:
        print("No feedback provided!")

    prompt = f"""
        {(example1_actors if mode == ShotPromptingMode.ONE_SHOT else f"{example1_actors}, {example2_actors}" if mode == ShotPromptingMode.FEW_SHOT else "")}

        Now extract the actors (roles of end users) from the following software
        description.

        **Description:**
        {project_description}

        **Output:**
    """

    return generate_response(prompt, sys_prompt, Actors)


def generate_high_level_goals(
    project_description,
    actors,
    feedback=None,
    mode=ShotPromptingMode.ZERO_SHOT,
    focused_request=False,
    generation_guidance: str | None = None,
):
    """
    Original HLG generator.

    Normal top-down calls keep ``focused_request=False`` and behave as before.
    Bottom-up correction calls use ``focused_request=True`` and may also pass
    ``generation_guidance`` produced by the Global Goal Evaluator. The guidance
    is a correction/scope hint, while ``project_description`` remains the
    factual source for the focused generation request.
    """
    sys_prompt = (
        "You are a helpful assistant expert in software engineering tasks. "
        "You're tasked to extract high-level goals from a software description "
        "for each provided actor that is expected to interact with the "
        "software. Following Goal-Oriented Requirements Engineering (GORE), "
        "high-level goals are strategic functional objectives that define the "
        "WHY behind a system. They are abstract, stakeholder-oriented, and "
        "independent of technical implementation. Generate ONLY functional "
        "goals."
    )

    if focused_request:
        # Bottom-up correction path only: the description has already been
        # narrowed to one intention, so pin the generator to exactly one HLG
        # instead of letting it derive a whole project-wide set again.
        sys_prompt += (
            "\n\nFOCUSED HLG GENERATION CONTRACT:\n"
            "- The supplied description represents exactly ONE autonomous "
            "functional stakeholder intention for the supplied actor.\n"
            "- Generate exactly ONE High-Level Goal for that intention.\n"
            "- Do not split operations, sub-capabilities, channels, views, "
            "settings, side effects, or secondary benefits into additional "
            "High-Level Goals.\n"
            "- Do not infer other project goals from the broader domain.\n"
            "- The goal must remain at WHY / stakeholder-objective abstraction "
            "level and be directly supported by the focused description.\n"
            "- Return a HighLevelGoals object containing exactly one goal.\n"
            "- Few-shot examples are style/GORE examples only; ignore their "
            "number of goals for this focused request.\n"
        )

    if generation_guidance is not None and generation_guidance.strip():
        sys_prompt += f"""

        BOTTOM-UP CORRECTION GUIDANCE:
        {generation_guidance.strip()}

        Use this only as guidance about the intended correction/scope. Do not
        copy it mechanically and do not introduce facts not supported by the
        supplied description.
        """

    if feedback is not None:
        print("Feedback provided!")
        sys_prompt += f"""

        The task was already attempted, but its output was evaluated as flawed.
        Apply the critique from the HLG evaluator to the previous attempt.
        Modify only what the critique identifies as needing correction.

        **Previous attempt:**
        {feedback.previous_output}

        **Critique:**
        {feedback.critique}
        """
    else:
        print("No feedback provided!")

    print("This is the provided sys prompt: ", sys_prompt)

    if focused_request:
        task_instruction = (
            "The description below has already been isolated to exactly one "
            "autonomous functional intention for the supplied actor. Generate "
            "exactly ONE High-Level Goal representing that WHY-level intention. "
            "Do not derive additional goals from operations, sub-capabilities, "
            "side effects, or secondary benefits."
        )
    else:
        task_instruction = (
            "Based on your understanding of the typical needs and interests of "
            "the following actors in the following software project, generate a "
            "list of high-level goals."
        )

    prompt = f"""
        {(example1_hl if mode == ShotPromptingMode.ONE_SHOT else f"{example1_hl}, {example2_hl}" if mode == ShotPromptingMode.FEW_SHOT else "")}

        Your task: {task_instruction}

        **Description:**
        {project_description}

        **Actors:**
        {actors}

        **Output:**
    """

    return generate_response(prompt, sys_prompt, HighLevelGoals)


def generate_evaluated_high_level_goals_from_request(
    request: HighLevelGoalGenerationRequest,
    mode=ShotPromptingMode.ZERO_SHOT,
    evaluator_ablation: bool = False,
) -> HighLevelGoals:
    """
    Bottom-up HLG correction path required by the thesis architecture:

        Global Goal Evaluator guidance
            -> ORIGINAL HLG generator
            -> ORIGINAL HLG evaluator/reflection loop
            -> evaluated focused HLG returned to the bottom-up cycle.

    The HLG evaluator is run in focused scope, so it judges completeness only
    for the single documented intention in this request, not for the whole
    project-wide HLG set.
    """
    if not isinstance(request, HighLevelGoalGenerationRequest):
        raise TypeError(
            "request must be a HighLevelGoalGenerationRequest instance."
        )

    # Lazy import keeps extractor.py usable by the original top-down pipeline
    # without creating an unnecessary module-level dependency cycle.
    from src.self_critique.refine_response import (
        EvalMode,
        generate_response_with_reflection,
    )

    def _focused_generator(*, feedback=None, mode=mode):
        return generate_high_level_goals(
            project_description=request.generator_input.project_description,
            actors=request.generator_input.actors,
            feedback=feedback,
            mode=mode,
            focused_request=True,
            generation_guidance=(
                request.generator_guidance or request.rationale
            ),
        )

    result, _, _ = generate_response_with_reflection(
        target_type="Focused High Level Goal",
        call_function=_focused_generator,
        define_args=[],
        eval_mode=EvalMode.HIGH_LEVEL,
        eval_args=[
            request.generator_input.project_description,
            request.generator_input.actors,
        ],
        shotPromptingMode=mode,
        llama_ablation=evaluator_ablation,
        focused_scope=True,
    )

    if not isinstance(result, HighLevelGoals):
        raise TypeError("The evaluated HLG pipeline did not return HighLevelGoals.")
    if len(result.goals) != 1:
        raise ValueError(
            "A focused bottom-up HLG request must return exactly one HLG after "
            f"its HLG evaluator loop; received {len(result.goals)}."
        )

    return result


def generate_low_level_goals(
    highLevelGoals,
    feedback=None,
    mode=ShotPromptingMode.ZERO_SHOT,
    generation_guidance: str | None = None,
):
    """
    Original LLG generator, decomposing HLGs into API-shaped low-level goals.

    ``generation_guidance`` is only ever set by the bottom-up selective
    regeneration path (see ``regenerate_evaluated_low_level_goals``); normal
    top-down calls leave it ``None``.
    """
    sys_prompt = (
        "You are a helpful assistant expert in software engineering tasks. "
        "Elicit low-level goals for a specific stakeholder in a software "
        "project. The low-level goals MUST be structured to match against API "
        "calls and must not be generic. Each low-level goal should be an atomic "
        "interaction with the system that could be implemented via an API call. "
        "Following GORE, low-level goals describe HOW high-level goals are "
        "achieved. Generate ONLY functional goals."
    )

    if generation_guidance is not None and generation_guidance.strip():
        # Bottom-up correction path only, see the docstring above.
        sys_prompt += f"""

        BOTTOM-UP DECOMPOSITION GUIDANCE:
        {generation_guidance.strip()}

        Use this guidance to correct the decomposition of the supplied HLGs.
        Do not introduce functions not supported by those HLGs and the project
        description used by the evaluator.
        """

    if feedback is not None:
        print("Feedback provided!")
        sys_prompt += f"""

        The task was already attempted, but its output was evaluated as flawed.
        Apply the critique from the LLG evaluator to the previous attempt.
        Modify only what the critique identifies as needing correction.

        **Previous attempt:**
        {feedback.previous_output}

        **Critique:**
        {feedback.critique}
        """
    else:
        print("No feedback provided!")

    print("This is the provided sys prompt: ", sys_prompt)

    prompt = f"""
        {(example1_ll if mode == ShotPromptingMode.ONE_SHOT else f"{example1_ll}, {example2_ll}" if mode == ShotPromptingMode.FEW_SHOT else "")}

        Your task: based on your understanding of the typical tasks that compose
        the following high-level goals, provide a decomposition into sub-goals.
        Each low-level goal should theoretically correspond to a single action
        of the actor with the software.

        **High-level goals:**
        {highLevelGoals}

        **Output:**
    """

    return generate_response(prompt, sys_prompt, LowLevelGoals)


def _actors_from_high_level_goals(high_level_goals: HighLevelGoals) -> Actors:
    """Preserve first-seen actor order while removing exact-name duplicates."""
    actors = []
    seen = set()
    for goal in high_level_goals.goals:
        # casefold() + collapsing whitespace catches case/spacing variants of
        # the same actor name (e.g. "Municipal Operator" vs "municipal  operator")
        # without pulling in a full semantic-similarity check for this.
        key = " ".join(goal.actor.name.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        actors.append(goal.actor)
    return Actors(actors=actors)


def regenerate_evaluated_low_level_goals(
    request: LowLevelGoalRegenerationRequest,
    *,
    project_description: str,
    mode=ShotPromptingMode.ZERO_SHOT,
    evaluator_ablation: bool = False,
) -> LowLevelGoals:
    """
    Selective LLG correction path:

        Global Goal Evaluator guidance
            -> ORIGINAL LLG generator
            -> ORIGINAL LLG evaluator/reflection loop
            -> evaluated LLG decomposition returned to the bottom-up cycle.
    """
    if not isinstance(request, LowLevelGoalRegenerationRequest):
        raise TypeError(
            "request must be a LowLevelGoalRegenerationRequest instance."
        )

    from src.self_critique.refine_response import (
        EvalMode,
        generate_response_with_reflection,
    )

    actors = _actors_from_high_level_goals(request.high_level_goals)
    guidance_parts = [
        f"- {parent_name}: {guidance}"
        for parent_name, guidance in request.guidance_by_parent_name.items()
    ]
    # request.guidance_by_parent_name only carries per-branch text when the
    # evaluator had something specific to say about that HLG; branches that
    # just need a fresh, otherwise-unguided decomposition fall back to a
    # generic instruction instead of an empty guidance string.
    combined_guidance = (
        "Regenerate only the supplied HLG branches.\n" + "\n".join(guidance_parts)
        if guidance_parts
        else "Regenerate a complete and coherent decomposition only for the supplied HLG branches."
    )

    def _focused_llg_generator(*, feedback=None, mode=mode):
        return generate_low_level_goals(
            request.high_level_goals,
            feedback=feedback,
            mode=mode,
            generation_guidance=combined_guidance,
        )

    result, _, _ = generate_response_with_reflection(
        target_type="Low Level Goals",
        call_function=_focused_llg_generator,
        define_args=[],
        eval_mode=EvalMode.LOW_LEVEL,
        eval_args=[
            project_description,
            actors,
            request.high_level_goals,
        ],
        shotPromptingMode=mode,
        llama_ablation=evaluator_ablation,
        focused_scope=True,
    )

    if not isinstance(result, LowLevelGoals):
        raise TypeError("The evaluated LLG pipeline did not return LowLevelGoals.")

    return result


def build_bottom_up_evaluated_generation_callbacks(
    *,
    project_description: str,
    mode=ShotPromptingMode.ZERO_SHOT,
    evaluator_ablation: bool = False,
):
    """
    Build the two callbacks expected by ``run_global_goal_cycle``.

    This is the safest notebook entry point because it guarantees that every
    HLG/LLG produced during bottom-up correction passes through the same
    generator-specific evaluator/reflection loop used by the original top-down
    pipeline. ``evaluator_ablation=True`` is an explicit *bottom-up correction*
    ablation and is deliberately independent from whether the saved baseline
    file was produced with the historical ``noLlama`` top-down condition.
    """

    def _generate_hlg(request: HighLevelGoalGenerationRequest) -> HighLevelGoals:
        return generate_evaluated_high_level_goals_from_request(
            request,
            mode=mode,
            evaluator_ablation=evaluator_ablation,
        )

    def _regenerate_llg(request: LowLevelGoalRegenerationRequest) -> LowLevelGoals:
        return regenerate_evaluated_low_level_goals(
            request,
            project_description=project_description,
            mode=mode,
            evaluator_ablation=evaluator_ablation,
        )

    return _generate_hlg, _regenerate_llg
