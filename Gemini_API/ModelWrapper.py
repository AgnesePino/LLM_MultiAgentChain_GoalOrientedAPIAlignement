"""Small Gemini wrapper with structured output and API-key rotation."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from google import genai
from google.genai import errors
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

from Gemini_API.APIKeysManager import APIKeysManager


MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_ATTEMPTS_PER_KEY = int(os.getenv("GEMINI_MAX_ATTEMPTS_PER_KEY", "2"))
MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "20000"))


class ModelWrapper:
    """Send independent Gemini requests using the configured key pool."""

    def __init__(self, sys_prompt: str, model_id: str = MODEL_ID):
        self._model_id = model_id
        self._sys_prompt = sys_prompt

    @staticmethod
    def _client():
        return genai.Client(api_key=APIKeysManager.get_api_key())

    def make_a_query(
        self,
        prompt: str,
        response_format: type[BaseModel] | None = None,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
    ) -> str | BaseModel:
        config: dict[str, Any] = {
            "system_instruction": self._sys_prompt,
            "temperature": 0,
            "max_output_tokens": max_output_tokens,
        }
        if response_format is not None:
            config.update(
                response_mime_type="application/json",
                response_schema=response_format,
            )

        attempts = max(1, APIKeysManager.count() * MAX_ATTEMPTS_PER_KEY)
        last_exception: Exception | None = None

        for attempt in range(attempts):
            client = None
            try:
                # Keep the Client alive for the complete HTTP request. Using
                # `self._client().models...` can release the temporary Client
                # immediately after resolving `.models`, closing its HTTP
                # transport before generate_content sends the request.
                client = self._client()
                response = client.models.generate_content(
                    model=self._model_id,
                    contents=prompt,
                    config=GenerateContentConfig(**config),
                )
                if not response.text:
                    raise RuntimeError("Gemini returned an empty response.")

                if response_format is None:
                    return response.text

                parsed = getattr(response, "parsed", None)
                if isinstance(parsed, response_format):
                    return parsed
                if parsed is not None:
                    return response_format.model_validate(parsed)
                return response_format.model_validate(json.loads(response.text))

            except errors.ClientError as exc:
                last_exception = exc
                if exc.code not in {400, 429}:
                    raise
                APIKeysManager.change_api_key()
            except errors.ServerError as exc:
                last_exception = exc
                if exc.code not in {500, 502, 503, 504}:
                    raise
                APIKeysManager.change_api_key()
            finally:
                if client is not None:
                    client.close()

            if attempt + 1 < attempts:
                time.sleep(min(0.5 * (attempt + 1), 2.0))

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("No Gemini API request could be completed.")


def generate_response(
    prompt: str,
    sys_prompt: str,
    response_format: type[BaseModel] | None = None,
) -> str | BaseModel:
    return ModelWrapper(sys_prompt=sys_prompt).make_a_query(
        prompt=prompt,
        response_format=response_format,
    )
