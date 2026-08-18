"""Utilities used by the original top-down pipeline."""

import json
from urllib.parse import urlparse

import requests

from src.data_model import API


def _normalize_github_url(link: str) -> str:
    """Convert common GitHub file/tree README links to raw text URLs."""
    parsed = urlparse(link)

    if parsed.netloc not in {"github.com", "www.github.com"}:
        return link

    parts = parsed.path.strip("/").split("/")
    if len(parts) < 4:
        return link

    owner, repository, mode, branch = parts[:4]
    remaining = parts[4:]

    if mode == "blob" and remaining:
        return (
            "https://raw.githubusercontent.com/"
            f"{owner}/{repository}/{branch}/{'/'.join(remaining)}"
        )

    if mode == "tree" and parsed.fragment.lower() == "readme":
        directory = "/".join(remaining)
        if directory:
            directory += "/"
        return (
            "https://raw.githubusercontent.com/"
            f"{owner}/{repository}/{branch}/{directory}README.md"
        )

    return link


def get_markdown(link: str):
    """Retrieve the README/documentation text."""
    response = requests.get(_normalize_github_url(link), timeout=30)
    response.raise_for_status()
    return response.text


def get_api_list_from_swagger(link):
    api_list = get_markdown(link)
    json_api_list = json.loads(api_list)["paths"]

    preprocessed_api_list = []

    for api_path, path in json_api_list.items():
        for method, operation in path.items():
            preprocessed_api_list.append(
                API(
                    api_name=operation["operationId"],
                    api_path=api_path,
                    description=operation["summary"],
                    request_type=method,
                )
            )

    return preprocessed_api_list


def api_list_to_string(api_list):
    return ", ".join(api.api_name for api in api_list) + "\n"
