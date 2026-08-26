"""
LLM clients shared by the original top-down pipeline and the bottom-up extension.

The top-down generator behaviour is kept equal to the original repository:
- gpt-4o-mini for structured Pydantic outputs;
- gpt-4o for plain-text outputs;
- llama-3.3-70b-versatile as the preferred evaluator.

The only infrastructure compatibility addition is a Qwen 3.6 fallback when
the historical Llama model is not available on the current Groq account.
"""

from __future__ import annotations

import os

from openai import APIStatusError, OpenAI, RateLimitError

from key import get_key_openai, get_key_llama, count_Llama_keys


OPENAI_STRUCTURED_MODEL = os.getenv("OPENAI_STRUCTURED_MODEL", "gpt-4o-mini")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o")

PREFERRED_GROQ_EVALUATOR_MODEL = os.getenv(
    "GROQ_EVALUATOR_MODEL",
    "llama-3.3-70b-versatile",
)
GROQ_EVALUATOR_FALLBACK_MODEL = os.getenv(
    "GROQ_EVALUATOR_FALLBACK_MODEL",
    "qwen/qwen3.6-27b",
)

GROQ_EVALUATOR_MODEL = PREFERRED_GROQ_EVALUATOR_MODEL

# Original repository budget for Llama.
GROQ_LLAMA_MAX_TOKENS = int(os.getenv("GROQ_LLAMA_MAX_TOKENS", "6000"))

# The fallback critic only has to produce Feedback + Score.
# 2000 also avoids the previous 900-token truncation while remaining much
# smaller than the original 6000-token reservation.
GROQ_FALLBACK_MAX_TOKENS = int(
    os.getenv("GROQ_FALLBACK_MAX_TOKENS", "2000")
)


client = OpenAI(api_key=get_key_openai())

llama = OpenAI(
    api_key=get_key_llama(),
    base_url="https://api.groq.com/openai/v1",
)


def _rotate_groq_key() -> None:
    global llama
    llama = OpenAI(
        api_key=get_key_llama(increment_counter=True),
        base_url="https://api.groq.com/openai/v1",
    )


def generate_response(prompt, sys_prompt, response_format=None):
    """Original GPT generator path."""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    if response_format is not None:
        response = client.beta.chat.completions.parse(
            messages=messages,
            model=OPENAI_STRUCTURED_MODEL,
            max_tokens=6000,
            response_format=response_format,
            temperature=0,
        )
        return response.choices[0].message.parsed

    response = client.chat.completions.create(
        messages=messages,
        model=OPENAI_TEXT_MODEL,
        max_tokens=6000,
        temperature=0,
    )
    return response.choices[0].message.content


def generate_response_llama(prompt, sys_prompt):
    """Evaluator call with the historical Llama model and one explicit fallback."""
    global GROQ_EVALUATOR_MODEL

    last_exception = None
    fallback_attempted = (
        GROQ_EVALUATOR_MODEL == GROQ_EVALUATOR_FALLBACK_MODEL
    )
    attempts = max(2, count_Llama_keys() * 2)

    for _ in range(attempts):
        model = GROQ_EVALUATOR_MODEL

        try:
            request = {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                "model": model,
                "temperature": 0,
            }

            if model == GROQ_EVALUATOR_FALLBACK_MODEL:
                request["max_completion_tokens"] = GROQ_FALLBACK_MAX_TOKENS
                request["extra_body"] = {
                    "reasoning_effort": "none",
                    "reasoning_format": "hidden",
                }
            else:
                request["max_completion_tokens"] = GROQ_LLAMA_MAX_TOKENS

            response = llama.chat.completions.create(**request)
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
