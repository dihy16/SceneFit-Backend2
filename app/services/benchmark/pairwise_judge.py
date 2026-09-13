"""Gemini-backed, blinded pairwise SceneFit judge."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, Field


PROMPT_VERSION = "scenefit-pairwise-elo-v2-fewshot"
DEFAULT_JUDGE_MODEL = "gemma-4-26b-a4b-it"
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
    reason: str = Field(max_length=160)


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
two outfits are genuinely equally suitable. Give one short reason of at most 20
words for each decision.

CALIBRATION EXAMPLES (text-only; do not copy their choices into the live batch):

Example A -- RIGHT should win overall:
Scene: a snowy outdoor commute. LEFT: a thin linen summer outfit. RIGHT: an
insulated coat and boots. Expected: RIGHT for climate_season,
activity_occasion, and overall. Judge style_theme and color_harmony separately
from warmth rather than automatically choosing RIGHT for every criterion.

Example B -- LEFT should win overall:
Scene: an elegant evening gallery opening. LEFT: understated tailored formalwear.
RIGHT: neon gym clothing. Expected: LEFT for activity_occasion, style_theme, and
overall. Climate_season may be a tie when the indoor climate is not evident.

Example C -- a genuine tie is appropriate:
Scene: a mild-weather casual park visit. LEFT and RIGHT are similarly casual,
weather-appropriate outfits with equally harmonious colors. Expected: TIE where
there is no meaningful compatibility difference, including overall.

Apply the rubric independently to the live image pairs below. The examples are
balanced deliberately and do not imply a preferred side.
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
        completion_instruction = self._completion_instruction(pairs)
        if self._types is None:
            return [
                RUBRIC,
                "SCENE:",
                scene_path,
                *[item for pair in pairs for item in pair],
                completion_instruction,
            ]
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
        parts.append(completion_instruction)
        return parts

    @staticmethod
    def _completion_instruction(
        pairs: Sequence[tuple[str, Path, Path]],
    ) -> str:
        pair_ids = [pair_id for pair_id, _, _ in pairs]
        numbered_ids = "\n".join(
            f"{index}. {pair_id}" for index, pair_id in enumerate(pair_ids, 1)
        )
        count = len(pair_ids)
        return f"""LIVE-BATCH COMPLETENESS CHECK

Return exactly {count} pair objects in the same order, one for each required ID:
{numbered_ids}

Before responding, verify that every required ID appears exactly once, no other
ID appears, and every pair contains all five criteria exactly once. Do not stop
early: the response is incomplete unless it contains all {count} pair objects.
"""

    def compare_batch(
        self,
        scene_path: Path,
        pairs: Sequence[tuple[str, Path, Path]],
    ) -> list[dict[str, Any]]:
        """Return one complete five-criterion decision for every pair ID."""
        expected_ids = [pair_id for pair_id, _, _ in pairs]
        pair_by_id = {
            pair_id: (pair_id, left_path, right_path)
            for pair_id, left_path, right_path in pairs
        }
        if len(pair_by_id) != len(expected_ids):
            raise ValueError("Pair request contains duplicate pair IDs")
        collected: dict[str, dict[str, Any]] = {}
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                pending_ids = [pair_id for pair_id in expected_ids if pair_id not in collected]
                pending_pairs = [pair_by_id[pair_id] for pair_id in pending_ids]
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
                        automatic_function_calling=self._types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self._contents(scene_path, pending_pairs),
                    config=config,
                )
                parsed = response.parsed
                if parsed is None:
                    parsed = PairDecisionBatch.model_validate_json(response.text)
                if isinstance(parsed, dict):
                    parsed = PairDecisionBatch.model_validate(parsed)
                received = {item.pair_id: item for item in parsed.pairs}
                if len(received) != len(parsed.pairs):
                    raise ValueError("Judge returned duplicate pair IDs")
                unexpected = set(received) - set(pending_ids)
                if unexpected:
                    raise ValueError(
                        f"Judge returned unexpected pair IDs {sorted(unexpected)}; "
                        f"pending IDs are {pending_ids}"
                    )
                for pair_id, decision in received.items():
                    by_criterion = {item.criterion: item for item in decision.decisions}
                    if set(by_criterion) != set(CRITERIA) or len(by_criterion) != len(decision.decisions):
                        raise ValueError(
                            f"Pair {pair_id} did not return exactly the required criteria"
                        )
                    collected[pair_id] = {
                        "pair_id": pair_id,
                        "decisions": [
                            by_criterion[criterion].model_dump()
                            for criterion in CRITERIA
                        ],
                    }
                missing_ids = [pair_id for pair_id in expected_ids if pair_id not in collected]
                if not missing_ids:
                    return [collected[pair_id] for pair_id in expected_ids]
                last_error = ValueError(
                    f"Judge omitted pair IDs {missing_ids}; received "
                    f"{sorted(received)} in this response"
                )
                print(
                    f"[PAIRWISE] Attempt {attempt + 1}/{self.max_attempts} returned "
                    f"{len(received)}/{len(pending_ids)} pending pairs; retrying only "
                    f"{missing_ids}",
                    flush=True,
                )
                if attempt + 1 < self.max_attempts:
                    self.sleep(self.base_delay * (2**attempt))
            except Exception as exc:
                last_error = exc
                print(
                    f"[PAIRWISE] Attempt {attempt + 1}/{self.max_attempts} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt + 1 < self.max_attempts:
                    self.sleep(self.base_delay * (2**attempt))
        missing_ids = [pair_id for pair_id in expected_ids if pair_id not in collected]
        raise RuntimeError(
            f"Pairwise Gemini judging failed with unresolved pair IDs {missing_ids}"
        ) from last_error
