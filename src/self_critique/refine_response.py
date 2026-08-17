from enum import Enum

import src.examples.shot_learning
from src.data_model import Critique
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
    """
    Run the original generator-specific evaluator.

    ``focused_scope=False`` preserves the original top-down behaviour: the
    generated collection is judged as the complete extraction for the supplied
    software description.

    ``focused_scope=True`` is used only by the bottom-up correction loop. In
    that case the supplied HLG(s) define the exact local scope being corrected,
    so the evaluator must not penalize the result because unrelated project
    goals/actors are intentionally absent from the selective regeneration.
    """
    if not isinstance(eval_mode, EvalMode):
        raise TypeError(
            f"Expected an instance of EvalMode, but got {type(eval_mode).__name__}"
        )

    sys_prompt = (
        "You're a helpful assistant, expert in software engineering and "
        "specialised in the Goal-Oriented Requirements Engineering (GORE) "
        "framework.\n\n"
        "Following GORE: "
        "- an actor is the active entity (the WHO); "
        "- high-level goals are strategic, functional WHY-level stakeholder "
        "objectives, independent of implementation; "
        "- low-level goals describe HOW high-level goals are achieved and are "
        "more concrete/operational.\n"
        "You may suggest corrections, but evaluate only functional goals.\n\n"
        "### Examples:\n\n"
    )

    if eval_mode == EvalMode.ACTORS:
        sys_prompt += f"""
            {src.examples.shot_learning.example1_actors_withFeedback1}

            ---

            {src.examples.shot_learning.example1_actors_withFeedback2}

            ---

            {src.examples.shot_learning.example1_actors_withFeedback3}

            ---

            {src.examples.shot_learning.example1_actors_withFeedback4}

            ---
        """
    elif eval_mode == EvalMode.HIGH_LEVEL:
        sys_prompt += f"""
            {src.examples.shot_learning.example1_hl_withFeedback1}

            ---

            {src.examples.shot_learning.example1_hl_withFeedback2}

            ---

            {src.examples.shot_learning.example1_hl_withFeedback3}

            ---

            {src.examples.shot_learning.example1_hl_withFeedback4}

            ---
        """
    elif eval_mode == EvalMode.LOW_LEVEL:
        sys_prompt += f"""
            {src.examples.shot_learning.example1_ll_withFeedback1}

            ---

            {src.examples.shot_learning.example1_ll_withFeedback2}

            ---

            {src.examples.shot_learning.example1_ll_withFeedback3}

            ---

            {src.examples.shot_learning.example1_ll_withFeedback4}

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
            "a software description and the actors (end user roles) for said "
            "software"
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
            "a software description, actors and high-level goals for said "
            "software"
        )
        assume_this_is_ok = "Assuming the work done on actors is ok,"
        critique_this = (
            "defining high-level end-user goals. Multiple actors can have the "
            "same goals (i.e., overlapping goals). Ensure that ONLY functional "
            "WHY-level goals are present"
        )
        additional_prompt = f"""
        **High-level goals:**
        {high_level_goals}
        """
    elif eval_mode == EvalMode.LOW_LEVEL:
        if low_level_goals is None or high_level_goals is None:
            raise ValueError(
                "EvalMode.LOW_LEVEL can only be used when both low_level_goals "
                "and high_level_goals are not None."
            )
        provided_with = (
            "a software description, actors, high-level goals and low-level "
            "goals for said software"
        )
        assume_this_is_ok = (
            "Assuming the work done on actors and high-level goals is ok,"
        )
        critique_this = (
            "defining low-level end-user goals. Each low-level goal should "
            "theoretically correspond to a single actor interaction with the "
            "software. Ensure that ONLY functional goals are present"
        )
        additional_prompt = f"""
        **High-level goals:**
        {high_level_goals}

        **Low-level goals:**
        {low_level_goals}
        """
    else:
        raise ValueError(f"Unsupported evaluation mode: {eval_mode!r}")

    if focused_scope:
        if eval_mode == EvalMode.HIGH_LEVEL:
            scope_rule = (
                "FOCUSED REGENERATION SCOPE: the supplied description has "
                "already been isolated to ONE autonomous functional intention "
                "for the supplied actor. Evaluate whether the generated output "
                "contains exactly ONE faithful WHY-level HLG for that intention. "
                "Do NOT penalize the output for omitting unrelated project HLGs. "
                "If more than one HLG is generated, or if the goal drifts to a "
                "secondary benefit/operation, treat that as a major error."
            )
        elif eval_mode == EvalMode.LOW_LEVEL:
            scope_rule = (
                "FOCUSED REGENERATION SCOPE: the supplied high-level goals are "
                "the ONLY parent goals being regenerated in this call. Evaluate "
                "completeness/coherence only with respect to those supplied HLGs. "
                "Do NOT penalize the output because actors, HLGs, or LLGs from "
                "other already-confirmed branches are intentionally absent."
            )
        else:
            scope_rule = (
                "FOCUSED REGENERATION SCOPE: evaluate only the explicitly "
                "supplied local scope and do not penalize intentionally omitted "
                "unrelated project elements."
            )
        completeness_rule = (
            "Completeness means complete for the focused scope above, not for "
            "the entire project."
        )
    else:
        scope_rule = ""
        completeness_rule = (
            f"The {provided_with} needs to be COMPLETELY representative of "
            "the software system described."
        )

    prompt = f"""
        You are provided with {provided_with}.
        These elements were extracted by another assistant from the software
        description.

        {scope_rule}

        {assume_this_is_ok} your job is to critique the work done by the
        assistant on {critique_this}.

        Give a score from 0 to 10 based on the critique you produced.
        Assign a score of 0 if you see any contradiction or important omission
        within the applicable scope. Assign the maximum score if you do not see
        any error.

        {completeness_rule}

        Decrease the score if something required by the applicable scope is
        missing, unsupported, contradictory, incorrectly abstracted, or in
        contrast with the description. In that case, give actionable feedback
        that the SAME generator can use on its next attempt. Otherwise, just
        produce the score.

        Respond exactly as:
        Feedback: [Feedback here]
        Score: [0.0-10.0]

        Do not add any other comments.

        ### Target Analysis
        **Description:**
        {description}

        **Actors:**
        {actors}

        {additional_prompt}

        ---
        ### Final Evaluation
        Generate the feedback and the score now.

        Feedback:
    """

    return generate_response_llama(prompt, sys_prompt)


def parse_evaluation(evaluation: str | Critique):
    if isinstance(evaluation, str):
        lines = [line.strip() for line in evaluation.strip().splitlines() if line.strip()]
        if not lines:
            raise ValueError("Empty evaluator output.")

        score_line = lines[-1]
        if not score_line.startswith("Score:"):
            raise ValueError("Input text does not contain a valid 'Score:' line.")

        feedback_line = " ".join(lines[:-1])
        if not feedback_line.startswith("Feedback:"):
            raise ValueError("Input text does not contain a valid 'Feedback:' line.")

        score = float(score_line.split(":", 1)[1].strip())
        feedback = feedback_line.split(":", 1)[1].strip()
    elif isinstance(evaluation, Critique):
        score = evaluation.score
        feedback = evaluation.comment
    else:
        raise ValueError("Input evaluation is not in the expected format.")

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
    """
    Run the original Generator -> Evaluator -> Feedback loop.

    The new ``focused_scope`` flag does not change normal top-down extraction;
    it only tells the existing evaluator that a bottom-up correction is
    intentionally regenerating a subset (one HLG or selected LLG branches).
    """
    feedback = None
    last_score = None
    last_critique = None

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
            return result, 10, None

        print(f"Evaluation for {target_type} STARTING...")
        evaluation = get_evaluation(
            eval_mode,
            *eval_args,
            result,
            focused_scope=focused_scope,
        )
        print(f"Evaluation for {target_type} DONE...")

        score, critique = parse_evaluation(evaluation)
        last_score = score
        last_critique = critique
        print(f"Score: {score}")
        print(f"Critique: {critique}")

        if score >= QUALITY_THRESHOLD:
            print("Satisfactory score achieved! Breaking out of the loop.")
            return result, score, critique

        print("Unsatisfactory score. Retrying...")
        feedback = Feedback(previous_output=result, critique=critique)

    print("Failed to achieve a satisfactory score within the maximum number of attempts.")
    return result, last_score, last_critique
