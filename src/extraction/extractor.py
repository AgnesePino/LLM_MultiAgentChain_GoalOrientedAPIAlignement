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
    if documentation_link is None:
        raise Exception("No documentation link provided")

    sys_prompt = (
        "You are a technical writing assistant specialized in summarizing software documentation. "
        "Your goal is to extract a clear, well-written, and accurate description of a project from its README file. "
        "The description should be natural and informative, without unnecessary details or implementation specifics. "
        "Avoid marketing language, vague claims, or filler content. "
        "Talk as you were a stakeholder describing the system he wants to be implemented (e.g., during requirements "
        "elicitation)."
    )

    prompt = (
        f"Here is the README file of a software project:\n\n{get_markdown(link=documentation_link)}\n\n"
        "Based on this README, write a well-structured description of the project. "
        "Explain its purpose, the problem it addresses (if mentioned), and its main functionalities. "
        "Do not include software implementation details"
    )

    return generate_response(prompt, sys_prompt, DocumentDescription)


def generate_actors(
    project_description,
    feedback=None,
    mode=ShotPromptingMode.ZERO_SHOT,
):
    sys_prompt = (
        "You are a helpful assistant expert in software engineering tasks, specialized in extracting end-users roles from a high level description of a software project. \n"
        "Your task is to extract the actors (roles of end users of the system) from the given description.\n"
        "If actors are not explicitly mentioned, infer them based on typical users of similar software systems."
        "Each extracted actor name should be accompanied by a very short description.\n"
    )

    if feedback is not None:
        print("Feedback provided!")
        sys_prompt += f"""

        The task given to you was already attempted but its output was flawed. You're provided with a critique on the previous attempt.
        The critique contains comments about actors, please take it into account when generating actors: with respect to the previous attempt,
        modify only what is mentioned in the critique.

        **Previous attempt:**
        {feedback.previous_output}

        **Critique:**
        {feedback.critique}
        """
    else:
        print("No feedback provided!")

    examples = (
        example1_actors
        if mode == ShotPromptingMode.ONE_SHOT
        else f"{example1_actors}, {example2_actors}"
        if mode == ShotPromptingMode.FEW_SHOT
        else ""
    )

    prompt = f"""
        {examples}\n

        Now extract the actors (roles of end users) from the following software description.

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
    *,
    focused_request: bool = False,
    generation_guidance: str | None = None,
):
    # Original repository prompt.
    sys_prompt = (
        "You are a helpful assistant expert in software engineering tasks."
        "You're tasked to extract high level goals from a software description for each provided actor that is expected to interact with the software."
        "Following the Goal-Oriented Requirements Engineering (GORE) frameworks, high-level goals are strategic objectives that define the 'why' behind a system. "
        "They are usually abstract, business-oriented, and independent of technical implementation. They represent the needs of stakeholders or the organization. "
        "Focus: Vision and justification. "
        "Generate ONLY the functional goals."
    )

    # Bottom-up-only extension; never used in the normal top-down call.
    if focused_request:
        sys_prompt += (
            "\n\nFOCUSED BOTTOM-UP REGENERATION:\n"
            "Generate exactly ONE High-Level Goal representing the single "
            "stakeholder intention described in the supplied focused input. "
            "Keep it at WHY level and do not infer unrelated goals."
        )

    if generation_guidance is not None and generation_guidance.strip():
        sys_prompt += (
            "\n\nBOTTOM-UP CORRECTION GUIDANCE:\n"
            f"{generation_guidance.strip()}\n"
            "Use the guidance only to correct the requested scope; do not "
            "introduce unsupported facts."
        )

    if feedback is not None:
        print("Feedback provided!")
        sys_prompt += f"""

        The task given to you was already attempted but its output was flawed. You're provided with a critique on the previous attempt.
        The critique contains comments about high level goals, please take it into account when generating high level goals: with respect to the previous attempt,
        modify only what is mentioned in the critique.

        **Previous attempt:**
        {feedback.previous_output}

        **Critique:**
        {feedback.critique}
        """
    else:
        print("No feedback provided!")

    print("This is the provided sys prompt: ", sys_prompt)

    examples = (
        example1_hl
        if mode == ShotPromptingMode.ONE_SHOT
        else f"{example1_hl}, {example2_hl}"
        if mode == ShotPromptingMode.FEW_SHOT
        else ""
    )

    if focused_request:
        task_instruction = (
            "generate exactly one high level goal for the single stakeholder "
            "intention represented by the following focused description."
        )
    else:
        task_instruction = (
            "based on your understanding of the typical needs and interests "
            "of the following actors in the following software project, "
            "generate a list of high level goals."
        )

    prompt = f"""
        {examples}\n

        \nYour task: {task_instruction}\n

        **Description:** \n\n
        {project_description}\n

        **Actors:**\n
        {actors}\n

        **Output:**
    """

    return generate_response(prompt, sys_prompt, HighLevelGoals)


def generate_low_level_goals(
    highLevelGoals,
    feedback=None,
    mode=ShotPromptingMode.ZERO_SHOT,
    *,
    generation_guidance: str | None = None,
):
    # Original repository prompt.
    sys_prompt = (
        "You are a helpful assistant expert in software engineering tasks. "
        "Elicit low-level goals for a specific stakeholder in a software project. "
        "The low-level goals that you create MUST be structured to match against a set of API calls. Don't be too generic, for example, avoid goals like 'make the software fast', 'develop a web interface' etc."
        "Each low-level goal MUST be phrased as an interaction with the system that could be implemented via an API call."
        "Avoid generic goals. Instead, break them down into atomic actions linked to system capabilities. "
        "Following the Goal-Oriented Requirements Engineering (GORE) framework, low-level goals are technical objectives "
        "that describe 'how' the high-level goals will be achieved. \n"
        "They are more concrete and are eventually refined into specific requirements or software specifications. \n"
        "Focus: Implementation and constraints. "
        "Generate ONLY the functional goals."
    )

    # Bottom-up-only extension; inert in normal top-down calls.
    if generation_guidance is not None and generation_guidance.strip():
        sys_prompt += (
            "\n\nBOTTOM-UP DECOMPOSITION GUIDANCE:\n"
            f"{generation_guidance.strip()}\n"
            "Regenerate only the supplied HLG scope and do not introduce "
            "unsupported functionality."
        )

    if feedback is not None:
        print("Feedback provided!")
        sys_prompt += f"""

        The task given to you was already attempted but its output was flawed. You're provided with a critique on the previous attempt.
        The critique contains comments about low-level goals, please take it into account when generating low-level goals: with respect to the previous attempt,
        modify only what is mentioned in the critique.

        **Previous attempt:**
        {feedback.previous_output}\n

        **Critique:**
        {feedback.critique}\n
        """
    else:
        print("No feedback provided!")

    print("This is the provided sys prompt: ", sys_prompt)

    examples = (
        example1_ll
        if mode == ShotPromptingMode.ONE_SHOT
        else f"{example1_ll}, {example2_ll}"
        if mode == ShotPromptingMode.FEW_SHOT
        else ""
    )

    prompt = f"""
        {examples}\n

        \nYour task: based on your understanding of the typical tasks that compose the following sequence of high-level goals,
        provide if possible a decomposition of goals into sub-goals.
        Each low-level goal should theoretically correspond to a single action of the actor with the software.

        **High-level goals:**\n\n
        {highLevelGoals}\n

        **Output:**
    """

    return generate_response(prompt, sys_prompt, LowLevelGoals)


# ---------------------------------------------------------------------------
# Bottom-up adapters
# The original top-down functions above remain the single generators.
# ---------------------------------------------------------------------------

def generate_evaluated_high_level_goals_from_request(
    request: HighLevelGoalGenerationRequest,
    mode=ShotPromptingMode.ZERO_SHOT,
    evaluator_ablation: bool = False,
) -> HighLevelGoals:
    if not isinstance(request, HighLevelGoalGenerationRequest):
        raise TypeError("request must be a HighLevelGoalGenerationRequest instance.")

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
        define_args=(),
        eval_mode=EvalMode.HIGH_LEVEL,
        eval_args=(
            request.generator_input.project_description,
            request.generator_input.actors,
        ),
        shotPromptingMode=mode,
        llama_ablation=evaluator_ablation,
        focused_scope=True,
    )

    if not isinstance(result, HighLevelGoals):
        raise TypeError("The evaluated HLG pipeline did not return HighLevelGoals.")

    if len(result.goals) != 1:
        raise ValueError(
            "A focused bottom-up HLG request must return exactly one HLG; "
            f"received {len(result.goals)}."
        )

    return result


def _actors_from_high_level_goals(high_level_goals: HighLevelGoals) -> Actors:
    actors = []
    seen = set()

    for goal in high_level_goals.goals:
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
    if not isinstance(request, LowLevelGoalRegenerationRequest):
        raise TypeError("request must be a LowLevelGoalRegenerationRequest instance.")

    from src.self_critique.refine_response import (
        EvalMode,
        generate_response_with_reflection,
    )

    actors = _actors_from_high_level_goals(request.high_level_goals)

    guidance_parts = [
        f"- {parent_name}: {guidance}"
        for parent_name, guidance in request.guidance_by_parent_name.items()
    ]
    guidance = (
        "Regenerate only the supplied HLG branches.\n" + "\n".join(guidance_parts)
        if guidance_parts
        else "Regenerate a complete decomposition only for the supplied HLG branches."
    )

    def _focused_generator(*, feedback=None, mode=mode):
        return generate_low_level_goals(
            request.high_level_goals,
            feedback=feedback,
            mode=mode,
            generation_guidance=guidance,
        )

    result, _, _ = generate_response_with_reflection(
        target_type="Low Level Goals",
        call_function=_focused_generator,
        define_args=(),
        eval_mode=EvalMode.LOW_LEVEL,
        eval_args=(
            project_description,
            actors,
            request.high_level_goals,
        ),
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
