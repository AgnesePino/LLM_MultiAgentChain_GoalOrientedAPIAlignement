"""Thread-safe rotation of Gemini API keys stored in environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv


def _load_environment() -> None:
    """Load Gemini variables without tying the code to a private filename."""
    selected_file = os.getenv("DOTENV_PATH")
    if selected_file:
        load_dotenv(dotenv_path=selected_file)
        return

    load_dotenv()
    if os.getenv("GOOGLE_API_KEY_1") or os.getenv("GOOGLE_API_KEY"):
        return

    # The notebooks run with `notebook/` as their working directory, while
    # the project historically keeps a named *.env file in the repository
    # root. Resolve that root from this module instead of relying on cwd.
    gemini_api_dir = Path(__file__).resolve().parent
    project_root = gemini_api_dir.parent
    search_roots = (
        gemini_api_dir,
        project_root,
        Path.cwd().resolve(),
    )

    # Gemini_API is authoritative. Fall back to the project root or cwd only
    # when the preceding location contains no env file.
    checked_roots: set[Path] = set()
    for root in search_roots:
        if root in checked_roots:
            continue
        checked_roots.add(root)
        candidates = list(root.glob("*.env"))
        if len(candidates) == 1:
            load_dotenv(dotenv_path=candidates[0])
            return


_load_environment()


class APIKeysManager:
    """Return configured GOOGLE_API_KEY_N values in round-robin order."""

    _lock = Lock()
    _index = 0

    @classmethod
    def _keys(cls) -> list[str]:
        raw_count = os.getenv("NUM_GOOGLE_API_KEYS", "").strip()
        if raw_count:
            try:
                count = int(raw_count)
            except ValueError as exc:
                raise ValueError("NUM_GOOGLE_API_KEYS must be an integer.") from exc
            if count < 1:
                raise ValueError("NUM_GOOGLE_API_KEYS must be at least 1.")
        else:
            count = 1

        keys = [
            value.strip()
            for number in range(1, count + 1)
            if (value := os.getenv(f"GOOGLE_API_KEY_{number}", "")).strip()
        ]

        # Also accept the standard SDK variable for a one-key setup.
        if not keys and (value := os.getenv("GOOGLE_API_KEY", "").strip()):
            keys = [value]

        if not keys:
            raise RuntimeError(
                "No Gemini key found. Set GOOGLE_API_KEY_1 (and optionally "
                "GOOGLE_API_KEY_2, ...) plus NUM_GOOGLE_API_KEYS."
            )
        return keys

    @classmethod
    def count(cls) -> int:
        return len(cls._keys())

    @classmethod
    def get_api_key(cls) -> str:
        keys = cls._keys()
        with cls._lock:
            cls._index %= len(keys)
            return keys[cls._index]

    @classmethod
    def change_api_key(cls) -> str:
        keys = cls._keys()
        with cls._lock:
            cls._index = (cls._index + 1) % len(keys)
            return keys[cls._index]

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._index = 0


# Compatibility export for older imports. It is callable so the value always
# reflects the current environment.
NUM_API_KEYS = APIKeysManager.count
