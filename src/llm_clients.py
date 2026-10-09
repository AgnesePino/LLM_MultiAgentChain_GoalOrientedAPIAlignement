"""
LLM clients shared by the top-down pipeline and the bottom-up extension.

Gemini generates descriptions, actors, HLGs, LLGs and other structured
generator outputs. Qwen 3.8 on Groq is the evaluator.

Historical Llama-oriented names are retained only for notebook compatibility.
"""

from __future__ import annotations

import os
from importlib import import_module

from Gemini_API.ModelWrapper import MODEL_ID as GEMINI_MODEL
from Gemini_API.ModelWrapper import generate_response as generate_gemini_response


# Compatibility aliases for notebooks that display these old attributes.
# generate_response no longer sends requests to OpenAI.
OPENAI_STRUCTURED_MODEL = GEMINI_MODEL
OPENAI_TEXT_MODEL = GEMINI_MODEL

# Keep the evaluator fixed so environment variables cannot silently select a
# different model and make experimental runs incomparable.
PREFERRED_GROQ_EVALUATOR_MODEL = "qwen/qwen3.8-27b"
GROQ_EVALUATOR_FALLBACK_MODEL = PREFERRED_GROQ_EVALUATOR_MODEL

GROQ_EVALUATOR_MODEL = PREFERRED_GROQ_EVALUATOR_MODEL

# The evaluator only has to produce ``Feedback`` and ``Score``.  Groq counts
# the requested completion budget together with the prompt against the
# organisation's tokens-per-minute ceiling. Reserving the historical 6000
# tokens therefore makes an otherwise small evaluator call exceed the
# available limit and Groq rejects it with HTTP 413.
GROQ_LLAMA_MAX_TOKENS = int(os.getenv("GROQ_LLAMA_MAX_TOKENS", "700"))

# Keep the fallback under the same request-size ceiling.
GROQ_FALLBACK_MAX_TOKENS = int(
    os.getenv("GROQ_FALLBACK_MAX_TOKENS", "700")
)

# The current evaluator prompt is always sent in full. Only compact memories
# from previous calls share this additional character budget.
GROQ_CRITIC_HISTORY_MAX_CHARS = int(
    os.getenv("GROQ_CRITIC_HISTORY_MAX_CHARS", "2500")
)


llama = None


class EvaluatorConversation:
    """Bounded, compact chat history for one bottom-up critic execution."""

    def __init__(self, max_turns: int = 8, max_state_updates: int = 8):
        if max_turns < 1 or max_state_updates < 1:
            raise ValueError("Conversation memory limits must be positive.")
        self.max_turns = max_turns
        self.max_state_updates = max_state_updates
        self._exchanges: list[tuple[str, str]] = []
        self._state_updates: list[str] = []

    def build_messages(self, prompt: str, sys_prompt: str) -> list[dict[str, str]]:
        system_content = sys_prompt
        if self._state_updates:
            system_content += (
                "\n\nMost recent bottom-up state supplied as explicit input "
                "for this evaluation (the previous iteration when available):\n"
                f"{self._state_updates[-1]}"
            )

        messages = [{"role": "system", "content": system_content}]

        # Select the newest compact exchanges that fit. Replaying full earlier
        # prompts would resend the project description and goal collections on
        # every critic call, quickly exceeding Groq's request/token limit.
        selected_exchanges: list[tuple[str, str]] = []
        retained_chars = 0
        for label, previous_response in reversed(self._exchanges):
            exchange_chars = len(label) + len(previous_response)
            if retained_chars + exchange_chars > GROQ_CRITIC_HISTORY_MAX_CHARS:
                continue
            selected_exchanges.append((label, previous_response))
            retained_chars += exchange_chars

        for label, previous_response in reversed(selected_exchanges):
            messages.extend([
                {"role": "user", "content": label},
                {"role": "assistant", "content": previous_response},
            ])
        messages.append({"role": "user", "content": prompt})
        return messages

    def record_exchange(self, label: str, response: str) -> None:
        normalized_label = " ".join(label.split())
        if not normalized_label:
            return
        self._exchanges.append((normalized_label, response))
        self._exchanges = self._exchanges[-self.max_turns:]

    def remember_state(self, update: str) -> None:
        normalized = " ".join(update.split())
        if not normalized:
            return
        self._state_updates.append(normalized)
        self._state_updates = self._state_updates[-self.max_state_updates:]

    @property
    def retained_turns(self) -> int:
        return len(self._exchanges)


def _key_helpers():
    """Load legacy Groq key helpers only when the evaluator is requested."""
    module = import_module("key")
    return module.get_key_llama, module.count_Llama_keys


def _groq_client():
    global llama
    if llama is None:
        from openai import OpenAI

        get_key_llama, _ = _key_helpers()
        llama = OpenAI(
            api_key=get_key_llama(),
            base_url="https://api.groq.com/openai/v1",
        )
    return llama


def _rotate_groq_key() -> None:
    global llama
    from openai import OpenAI

    get_key_llama, _ = _key_helpers()
    llama = OpenAI(
        api_key=get_key_llama(increment_counter=True),
        base_url="https://api.groq.com/openai/v1",
    )


def generate_response(prompt, sys_prompt, response_format=None):
    """Generate text or a validated Pydantic object with Gemini."""
    return generate_gemini_response(
        prompt=prompt,
        sys_prompt=sys_prompt,
        response_format=response_format,
    )


def generate_evaluator_response(
    prompt,
    sys_prompt,
    conversation: EvaluatorConversation | None = None,
    temperature: float = 0,
):
    """Run Qwen on Groq, optionally replaying a bounded conversation."""
    global GROQ_EVALUATOR_MODEL

    from openai import APIStatusError, RateLimitError

    last_exception = None
    fallback_attempted = (
        GROQ_EVALUATOR_MODEL == GROQ_EVALUATOR_FALLBACK_MODEL
    )
    # Try each configured key at most once per evaluator request. Retrying
    # every key twice can block one dataset for many minutes when the whole
    # account is rate-limited, preventing the outer dataset loop from moving on.
    _, count_llama_keys = _key_helpers()
    attempts = max(1, count_llama_keys())

    for _ in range(attempts):
        model = GROQ_EVALUATOR_MODEL

        try:
            groq_client = _groq_client()
            request = {
                "messages": (
                    conversation.build_messages(prompt, sys_prompt)
                    if conversation is not None
                    else [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ]
                ),
                "model": model,
                "temperature": temperature,
            }

            if model == GROQ_EVALUATOR_FALLBACK_MODEL:
                request["max_completion_tokens"] = GROQ_FALLBACK_MAX_TOKENS
            else:
                request["max_completion_tokens"] = GROQ_LLAMA_MAX_TOKENS

            # Reasoning controls are model-specific, not dependent on whether
            # the model happens to be configured as primary or fallback.
            if model.startswith("qwen/"):
                request["extra_body"] = {
                    "reasoning_effort": "none",
                    "reasoning_format": "hidden",
                }
            elif model.startswith("openai/gpt-oss"):
                request["extra_body"] = {
                    "reasoning_effort": "low",
                    "reasoning_format": "hidden",
                }

            response = groq_client.chat.completions.create(**request)
            content = response.choices[0].message.content

            if not content:
                raise RuntimeError("The evaluator returned an empty response.")

            return content

        except RateLimitError as exc:
            last_exception = exc
            _rotate_groq_key()

        except APIStatusError as exc:
            last_exception = exc

            if (
                exc.status_code == 404
                and not fallback_attempted
                and model != GROQ_EVALUATOR_FALLBACK_MODEL
            ):
                GROQ_EVALUATOR_MODEL = GROQ_EVALUATOR_FALLBACK_MODEL
                fallback_attempted = True
                print(
                    "Groq evaluator model not available; using fallback: "
                    f"{GROQ_EVALUATOR_MODEL}"
                )
                continue

            raise

    if last_exception is not None:
        raise last_exception

    raise RuntimeError("No Groq evaluator API call could be completed.")


# Compatibility alias for older notebooks and external imports.
generate_response_llama = generate_evaluator_response
