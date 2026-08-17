"""
llm_clients.py

Utility functions for the two LLM providers used by the project:
- OpenAI for the generators/reconstructors;
- Groq for the evaluator/critic role.

The historical function name ``generate_response_llama`` is kept for backward
compatibility with the existing project imports, even though the configured
Groq evaluator model is now selected through ``GROQ_EVALUATOR_MODEL``.
"""
from openai import OpenAI, RateLimitError
from key import get_key_openai, get_key_llama, count_Llama_keys


# How many times a caller may re-ask a generator for a *semantically* better
# answer (e.g. a missing branch, a rejected duplicate). Unrelated to the
# RateLimitError key-rotation retries in generate_response_evaluator below.
MAX_SEMANTIC_RETRIES = 3

# Groq retired llama-3.3-70b-versatile for Free/Developer usage on 2026-08-16.
# Keep the evaluator model in one place so an infrastructure change never
# requires editing every evaluator call site.
GROQ_EVALUATOR_MODEL = "openai/gpt-oss-120b"


client = OpenAI(api_key=get_key_openai())

# The existing key helpers keep their historical "llama" names because they
# actually manage Groq API keys in this project.
llama = OpenAI(
    api_key=get_key_llama(),
    base_url="https://api.groq.com/openai/v1",
)


def generate_response(prompt, sys_prompt, response_format=None):
    """Call OpenAI. With a Pydantic ``response_format`` it returns a parsed
    structured object; without one it returns the raw text content."""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    if response_format is not None:
        response = client.beta.chat.completions.parse(
            messages=messages,
            model="gpt-4o-mini",
            max_tokens=6000,
            response_format=response_format,
            temperature=0,
        )
        return response.choices[0].message.parsed

    response = client.beta.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        max_tokens=6000,
        temperature=0,
    )
    return response.choices[0].message.content


def generate_response_evaluator(prompt, sys_prompt):
    """Call the Groq-backed evaluator, rotating Groq keys on rate limiting."""
    global llama
    last_exception = None

    # x2 so every configured key gets a second chance after the rotation has
    # gone all the way around once, instead of giving up after a single pass.
    for _ in range(count_Llama_keys() * 2):
        try:
            response = llama.beta.chat.completions.parse(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=GROQ_EVALUATOR_MODEL,
                max_tokens=6000,
                temperature=0,
            )
            return response.choices[0].message.content

        except RateLimitError as exc:
            last_exception = exc
            llama = OpenAI(
                api_key=get_key_llama(increment_counter=True),
                base_url="https://api.groq.com/openai/v1",
            )

    if last_exception is None:
        # Only reachable if count_Llama_keys() returns 0, i.e. no key was
        # ever configured, so the loop body above never ran.
        raise RuntimeError("No Groq evaluator API key is available.")
    raise last_exception


# Backward-compatible name used throughout the existing codebase.
def generate_response_llama(prompt, sys_prompt):
    return generate_response_evaluator(prompt, sys_prompt)
