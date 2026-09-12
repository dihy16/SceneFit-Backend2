"""Gemini-backed, blinded pairwise SceneFit judge."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, Field


PROMPT_VERSION = "scenefit-pairwise-elo-v1"
DEFAULT_JUDGE_MODEL = "gemma-4-31b-it"
CRITERIA = (
    "climate_season",
    "activity_occasion",
    "style_theme",
    "color_harmony",
    "overall",
)


class CriterionDecision(BaseModel):
    criterion: Literal[
        "climate_season",
        "activity_occasion",
        "style_theme",
        "color_harmony",
        "overall",
    ]
    winner: Literal["left", "right", "tie"]
    reason: str = Field(max_length=240)


class PairDecision(BaseModel):
    pair_id: str
    decisions: list[CriterionDecision]


class PairDecisionBatch(BaseModel):
    pairs: list[PairDecision]


RUBRIC = """You are an independent fashion-retrieval evaluator.
The first image is the SCENE. Each numbered pair after it contains a LEFT and RIGHT
outfit. Judge only the visual compatibility of each outfit with this scene; do not
infer identity from filenames or labels.

For each pair, independently choose left, right, or tie for these criteria:
- climate_season: suitability for the apparent weather, temperature, and season;
- activity_occasion: suitability for the apparent activity, setting, and formality;
- style_theme: harmony with the scene's visual/cultural theme;
- color_harmony: color harmony with the scene;
- overall: overall outfit suitability for this scene.

Ignore image quality, pose, body shape, and presentation. Use tie only when the
two outfits are genuinely equally suitable. Return exactly one decision for every
criterion of every supplied pair ID.
"""


class PairwiseGeminiJudge:
    """Structured Gemini pairwise judge with bounded retry/backoff."""

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
            raise ValueError("GEMINI_API_KEY is required for pairwise Gemini judging")
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
        pairs: Sequence[tuple[str, Path, Path]],
    ) -> list[Any]:
        if self._types is None:
            return [RUBRIC, scene_path, *[item for pair in pairs for item in pair]]
        parts: list[Any] = [RUBRIC, "SCENE:"]
        parts.append(
            self._types.Part.from_bytes(
                data=scene_path.read_bytes(), mime_type=self._mime_type(scene_path)
            )
        )
        for pair_id, left_path, right_path in pairs:
            parts.extend((f"PAIR ID: {pair_id}; LEFT OUTFIT:",))
            parts.append(
                self._types.Part.from_bytes(
                    data=left_path.read_bytes(), mime_type=self._mime_type(left_path)
                )
            )
            parts.append("RIGHT OUTFIT:")
            parts.append(
                self._types.Part.from_bytes(
                    data=right_path.read_bytes(), mime_type=self._mime_type(right_path)
                )
            )
        return parts

    def compare_batch(
        self,
        scene_path: Path,
        pairs: Sequence[tuple[str, Path, Path]],
    ) -> list[dict[str, Any]]:
        """Return one complete five-criterion decision for every pair ID."""
        expected_ids = [pair_id for pair_id, _, _ in pairs]
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                config: Any = {
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_schema": PairDecisionBatch,
                }
                if self._types is not None:
                    config = self._types.GenerateContentConfig(
                        temperature=0,
                        thinking_config=self._types.ThinkingConfig(thinking_level="minimal"),
                        response_mime_type="application/json",
                        response_schema=PairDecisionBatch,
                    )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self._contents(scene_path, pairs),
                    config=config,
                )
                parsed = response.parsed
                if parsed is None:
                    parsed = PairDecisionBatch.model_validate_json(response.text)
                if isinstance(parsed, dict):
                    parsed = PairDecisionBatch.model_validate(parsed)
                received = {item.pair_id: item for item in parsed.pairs}
                if set(received) != set(expected_ids) or len(received) != len(parsed.pairs):
                    raise ValueError(
                        f"Judge returned pair IDs {sorted(received)}; expected {expected_ids}"
                    )
                result: list[dict[str, Any]] = []
                for pair_id in expected_ids:
                    decision = received[pair_id]
                    by_criterion = {item.criterion: item for item in decision.decisions}
                    if set(by_criterion) != set(CRITERIA) or len(by_criterion) != len(decision.decisions):
                        raise ValueError(
                            f"Pair {pair_id} did not return exactly the required criteria"
                        )
                    result.append(
                        {
                            "pair_id": pair_id,
                            "decisions": [
                                by_criterion[criterion].model_dump()
                                for criterion in CRITERIA
                            ],
                        }
                    )
                return result
            except Exception as exc:
                last_error = exc
                print(
                    f"[PAIRWISE] Attempt {attempt + 1}/{self.max_attempts} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt + 1 < self.max_attempts:
                    self.sleep(self.base_delay * (2**attempt))
        raise RuntimeError("Pairwise Gemini judging failed") from last_error
