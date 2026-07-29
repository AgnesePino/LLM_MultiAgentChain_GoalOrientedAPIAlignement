
"""
llm_client.py

This module provides utility functions to interact with different LLM APIs, 
such as OpenAI's GPT-4 and Meta's LLaMA. It includes functions to generate 
responses using various models, handling system prompts and formatting.

API clients for OpenAI and LLaMA are initialized at the beginning.
"""
import groq
from openai import OpenAI, RateLimitError
from key import get_key_openai, get_key_llama, count_Llama_keys

# Set your GPT-4 API key
client = OpenAI(
    api_key= get_key_openai()
)

# Set your llama API key, still using the OpenAI client API
llama = OpenAI(
    api_key=get_key_llama(),
    base_url = "https://api.groq.com/openai/v1"
)

def generate_response(prompt, sys_prompt, response_format=None) -> str:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]

    if response_format is not None:
        response = client.beta.chat.completions.parse(
            messages=messages,
            #model="gpt-4o",
            model="gpt-4o-mini",
            max_tokens=6000,
            response_format=response_format,
            temperature=0
        )
        return response.choices[0].message.parsed
    else:
        response = client.beta.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            #model="gpt-4o-mini",
            max_tokens=6000,
            temperature=0
        )
        return response.choices[0].message.content



def generate_response_llama(prompt, sys_prompt):
    global llama
    last_exception = None

    for _ in range(0, count_Llama_keys()*2):
        try:
            response = llama.beta.chat.completions.parse(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                model = "llama-3.3-70b-versatile",
                    # model="llama3.1-8b",
                max_tokens = 6000,
                temperature = 0
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            last_exception = e

            llama = OpenAI(
                api_key=get_key_llama(increment_counter=True),
                base_url="https://api.groq.com/openai/v1"
            )

    raise last_exception


    



