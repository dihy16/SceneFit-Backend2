from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field


PROMPT_VERSION = "scene-outfit-compatibility-v1"
DEFAULT_JUDGE_MODEL = "gemma-4-31b-it"


class OutfitJudgment(BaseModel):
    outfit_id: str
    score: int = Field(ge=1, le=5)
    reason: str


class JudgmentBatch(BaseModel):
    judgments: list[OutfitJudgment]


RUBRIC = """You are an independent fashion retrieval evaluator.
The first image is the SCENE. Every following image is an OUTFIT and is preceded by its exact outfit ID.
Score each outfit's compatibility with the scene independently. Consider climate/season, activity or
occasion, style/theme, and color harmony. Ignore image quality, pose, body shape, and presentation.

Use this integer rubric:
5 = excellent, highly suitable match
4 = good match with only minor issues
3 = acceptable or neutral match
2 = poor match with substantial incompatibility
1 = clearly unsuitable match

Return exactly one judgment for every supplied outfit ID. Do not rename IDs.
"""


class GeminiJudge:
    """Gemini API-hosted multimodal evaluator with retry/backoff."""

    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_JUDGE_MODEL,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.sleep = sleep
        if client is not None:
            self.client = client
            self._types = None
            return
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required for Gemini judging")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install google-genai before running Gemini judging") from exc
        self.client = genai.Client(api_key=key)
        self._types = types

    @staticmethod
    def _mime_type(path: Path) -> str:
        suffix = path.suffix.lower()
        return "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix.lstrip('.')}"

    def _contents(
        self,
        scene_path: Path,
        outfits: Sequence[tuple[str, Path]],
    ) -> list[Any]:
        if self._types is None:
            return [RUBRIC, scene_path, *[value for pair in outfits for value in pair]]
        parts: list[Any] = [RUBRIC, "SCENE:"]
        parts.append(
            self._types.Part.from_bytes(
                data=scene_path.read_bytes(),
                mime_type=self._mime_type(scene_path),
            )
        )
        for outfit_id, outfit_path in outfits:
            parts.append(f"OUTFIT ID: {outfit_id}")
            parts.append(
                self._types.Part.from_bytes(
                    data=outfit_path.read_bytes(),
                    mime_type=self._mime_type(outfit_path),
                )
            )
        return parts

    def score_batch(
        self,
        scene_path: Path,
        outfits: Sequence[tuple[str, Path]],
    ) -> list[dict[str, Any]]:
        return self._score_batch(scene_path, outfits, allow_repair=True)

    def _score_batch(
        self,
        scene_path: Path,
        outfits: Sequence[tuple[str, Path]],
        *,
        allow_repair: bool,
    ) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                config: Any = {
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_schema": JudgmentBatch,
                }
                if self._types is not None:
                    config = self._types.GenerateContentConfig(
                        temperature=0,
                        thinking_config=self._types.ThinkingConfig(
                            thinking_level="minimal"
                        ),
                        response_mime_type="application/json",
                        response_schema=JudgmentBatch,
                    )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self._contents(scene_path, outfits),
                    config=config,
                )
                parsed = response.parsed
                if parsed is None:
                    parsed = JudgmentBatch.model_validate_json(response.text)
                if isinstance(parsed, dict):
                    parsed = JudgmentBatch.model_validate(parsed)
                results = [item.model_dump() for item in parsed.judgments]
                expected_ids = [outfit_id for outfit_id, _ in outfits]
                returned_ids = [str(item["outfit_id"]) for item in results]
                if len(returned_ids) != len(expected_ids):
                    print(
                        "[GEMINI] Response row count differs from the request; "
                        "normalizing valid rows",
                        flush=True,
                    )
                first_by_id: dict[str, dict[str, Any]] = {}
                for result in results:
                    outfit_id = str(result["outfit_id"])
                    if outfit_id in expected_ids:
                        first_by_id.setdefault(outfit_id, result)
                missing_ids = [
                    outfit_id for outfit_id in expected_ids if outfit_id not in first_by_id
                ]
                if missing_ids and not allow_repair:
                    raise ValueError(
                        "Judge omitted requested outfit IDs: "
                        f"{missing_ids}; returned {returned_ids}"
                    )
                if missing_ids:
                    print(
                        f"[GEMINI] Repairing {len(missing_ids)} omitted outfit(s) "
                        "with individual requests",
                        flush=True,
                    )
                    outfit_paths = dict(outfits)
                    for outfit_id in missing_ids:
                        repaired = self._score_batch(
                            scene_path,
                            [(outfit_id, outfit_paths[outfit_id])],
                            allow_repair=False,
                        )
                        first_by_id[outfit_id] = repaired[0]
                return [first_by_id[outfit_id] for outfit_id in expected_ids]
            except Exception as exc:
                last_error = exc
                print(
                    f"[GEMINI] Attempt {attempt + 1}/{self.max_attempts} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt + 1 < self.max_attempts:
                    delay = self.base_delay * (2**attempt)
                    print(f"[GEMINI] Retrying in {delay:.1f}s", flush=True)
                    self.sleep(delay)
        raise RuntimeError(f"Gemini judging failed after {self.max_attempts} attempts") from last_error
