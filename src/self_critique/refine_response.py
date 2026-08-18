import re
from enum import Enum

from src.data_model import Critique
from src.examples import shot_learning
from src.examples.shot_learning import ShotPromptingMode
from src.llm_clients import generate_response_llama


MAX_ATTEMPTS = 3
QUALITY_THRESHOLD = 8.5


class EvalMode(Enum):
    ACTORS = "Actors"
    HIGH_LEVEL = "High Level Goals"
    LOW_LEVEL = "Low Level Goals"


class Feedback:
    def __init__(self, previous_output, critique):
        self.previous_output = previous_output
        self.critique = critique


def get_evaluation(
    eval_mode: EvalMode,
    description,
    actors,
    high_level_goals=None,
    low_level_goals=None,
    *,
    focused_scope: bool = False,
):
    if not isinstance(eval_mode, EvalMode):
        raise TypeError(
            f"Expected an instance of EvalMode, but got {type(eval_mode).__name__}"
        )

    # Original evaluator system prompt.
    sys_prompt = (
        "You're an helpful assistant, expert in the field of software engineering and specialised in the Goal-Oriented Requirements Engineering (GORE) framework.\n\n."
        "Following the Goal-Oriented Requirements Engineering (GORE) framework: "
        "-  an actor is active entity that has the capability to perform actions to achieve goals. Unlike goals, which are 'what' or 'why,'"
        " actors are the 'who.' \n"
        "-  high-level goals are strategic objectives that define the 'why' behind a system. \n"
        "   They are usually abstract, business-oriented, and independent of technical implementation. "
        "They represent the needs of stakeholders or the organization. \n"
        "   Focus: Vision and justification. \n"
        "-  low-level goals are technical objectives that describe 'how' the high-level goals will be achieved. \n"
        "   They are more concrete and are eventually refined into specific requirements or software specifications. \n"
        "   Focus: Implementation and constraints.\n"
        "You can propose new goals taking into account the already present ones. Consider that high-level "
        "goals often answer the WHY question, while low-level goals often address the HOW."
        "\n\n ### Examples:\n\n"
    )

    if eval_mode == EvalMode.ACTORS:
        sys_prompt += f"""
            {shot_learning.example1_actors_withFeedback1}

            ---

            {shot_learning.example1_actors_withFeedback2}

            ---

            {shot_learning.example1_actors_withFeedback3}

            ---

            {shot_learning.example1_actors_withFeedback4}

            ---
        """

    elif eval_mode == EvalMode.HIGH_LEVEL:
        sys_prompt += f"""
            {shot_learning.example1_hl_withFeedback1}

            ---

            {shot_learning.example1_hl_withFeedback2}

            ---

            {shot_learning.example1_hl_withFeedback3}

            ---

            {shot_learning.example1_hl_withFeedback4}

            ---
        """

    elif eval_mode == EvalMode.LOW_LEVEL:
        sys_prompt += f"""
            {shot_learning.example1_ll_withFeedback1}

            ---

            {shot_learning.example1_ll_withFeedback2}

            ---

            {shot_learning.example1_ll_withFeedback3}

            ---

            {shot_learning.example1_ll_withFeedback4}

            ---
        """

    additional_prompt = ""

    if eval_mode == EvalMode.ACTORS:
        if high_level_goals is not None or low_level_goals is not None:
            raise ValueError(
                "EvalMode.ACTORS can only be used when high_level_goals and "
                "low_level_goals are both None."
            )
        provided_with = (
            "a software description and the actors (end user roles) for said software"
        )
        assume_this_is_ok = (
            "Considering only the actors' names and not their descriptions,"
        )
        critique_this = "defining actors"

    elif eval_mode == EvalMode.HIGH_LEVEL:
        if low_level_goals is not None or high_level_goals is None:
            raise ValueError(
                "EvalMode.HIGH_LEVEL can only be used when low_level_goals is "
                "None and high_level_goals is not None."
            )
        provided_with = (
            "a software description, actors and high-level goals for said software. "
        )
        assume_this_is_ok = "Assuming the work done on actors is ok,"
        critique_this = (
            "defining high-level end users goals. Multiple actors can have the "
            "same goals (i.e., overlapping goals). You must ensure that ONLY "
            "functional goals are present."
        )
        additional_prompt = f"""
        **High-level goals:**\n\n
        {high_level_goals}
        """

    elif eval_mode == EvalMode.LOW_LEVEL:
        if low_level_goals is None or high_level_goals is None:
            raise ValueError(
                "EvalMode.LOW_LEVEL can only be used when both low_level_goals "
                "and high_level_goals are not None."
            )
        provided_with = (
            "a software description, actors, high-level goals and low-level goals "
            "for said software"
        )
        assume_this_is_ok = (
            "Assuming the work done on actors and high-level goals is ok,"
        )
        critique_this = (
            "defining low-level end users goals. Each low-level goal should "
            "theoretically correspond to a single action of the actor with the "
            "software. Multiple actors can have the same goals (i.e., overlapping "
            "goals). You must ensure that ONLY functional goals are present."
        )
        additional_prompt = f"""
        **High-level goals:**\n\n
        {high_level_goals}

        **Low-level goals:**\n\n
        {low_level_goals}
        """

    else:
        raise ValueError(f"Unsupported evaluation mode: {eval_mode!r}")

    # Bottom-up-only scope rule. Empty during the original top-down execution.
    focused_scope_prompt = ""
    if focused_scope:
        if eval_mode == EvalMode.HIGH_LEVEL:
            focused_scope_prompt = (
                "FOCUSED BOTTOM-UP SCOPE: evaluate completeness only for the "
                "single supplied HLG intention. Do not penalize unrelated "
                "project goals that are intentionally absent."
            )
        elif eval_mode == EvalMode.LOW_LEVEL:
            focused_scope_prompt = (
                "FOCUSED BOTTOM-UP SCOPE: evaluate completeness only for the "
                "supplied HLG branch(es). Do not penalize LLGs from unrelated "
                "branches that are intentionally absent."
            )

    prompt = f"""
        You are provided with {provided_with}.\n
        These informations were extracted by another assistant from the software description.\n
        {focused_scope_prompt}
        {assume_this_is_ok} your job is to critique the work done by the assistant on {critique_this}.
        Give a score from 0 to 10 based on the critique you produced.
        Assign a score of 0 if you see any contradiction or important omissions.
        Assign the maximum score if you don't see any error.
        The {provided_with} needs to be COMPLETELY representative of the software system described.
        As in the previous examples, decrease the score if you think that something may be missing,
        or if there is something in contrast with the system's description.
        In that case, suggest what to do to improve the {provided_with}.
        Otherwise, just produce the score.

        Respond with exactly these two fields, in this order.
        The following is an example of the FORMAT only; replace its values with your real evaluation:

        Feedback: The extraction is coherent but one relevant functional goal is missing.
        Score: 8.0

        The value after "Score:" MUST be a numeric value from 0.0 to 10.0.
        Never output placeholders such as [0-10], [0.0-10.0] or [X.X].
        Do not add any other comments.

        ### Target Analysis
        **Description:**
        {description}

        **Actors:**
        {actors}

        {additional_prompt}

        ---
        ### Final Evaluation
        Generate the feedback and numeric score now.
    """

    return generate_response_llama(prompt, sys_prompt)


def parse_evaluation(evaluation: str | Critique):
    if isinstance(evaluation, Critique):
        return float(evaluation.score), evaluation.comment

    if not isinstance(evaluation, str):
        raise ValueError("Input evaluation is not in the expected format.")

    score_match = re.search(
        r"(?im)^\s*Score:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/10)?\s*$",
        evaluation,
    )
    feedback_match = re.search(
        r"(?ims)^\s*Feedback:\s*(.*?)(?=^\s*Score:|\Z)",
        evaluation,
    )

    if score_match is None:
        raise ValueError(
            "Input text does not contain a numeric 'Score:' line. "
            f"Raw evaluator output: {evaluation!r}"
        )

    if feedback_match is None:
        raise ValueError(
            "Input text does not contain a valid 'Feedback:' field. "
            f"Raw evaluator output: {evaluation!r}"
        )

    score = float(score_match.group(1))
    feedback = " ".join(feedback_match.group(1).split())

    return score, feedback


def generate_response_with_reflection(
    target_type,
    call_function,
    define_args,
    eval_mode,
    eval_args,
    shotPromptingMode=ShotPromptingMode.ZERO_SHOT,
    max_attempts=MAX_ATTEMPTS,
    llama_ablation=False,
    *,
    focused_scope: bool = False,
):
    feedback = None
    result = None
    score = None
    critique = None

    for attempt in range(1, max_attempts + 1):
        print(f"{target_type} STARTING... (attempt {attempt})")

        result = call_function(
            *define_args,
            feedback=feedback,
            mode=shotPromptingMode,
        )

        print(f"{target_type} DONE...")
        print(result)

        if llama_ablation:
            return result, 10.0, None

        print(f"Evaluation for {target_type} STARTING...")

        evaluation = get_evaluation(
            eval_mode,
            *eval_args,
            result,
            focused_scope=focused_scope,
        )

        print(f"Evaluation for {target_type} DONE...")

        score, critique = parse_evaluation(evaluation)

        print(f"Score: {score}")
        print(f"Critique: {critique}")

        if score >= QUALITY_THRESHOLD:
            print("Satisfactory score achieved! Breaking out of the loop.")
            return result, score, critique

        print("Unsatisfactory score. Retrying...")
        feedback = Feedback(
            previous_output=result,
            critique=critique,
        )

    # Original experimental policy: retain the last generated response.
    print(
        "Failed to achieve a satisfactory score within the maximum number "
        "of attempts. Returning the last generated result."
    )
    return result, score, critique
