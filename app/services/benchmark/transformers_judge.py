"""In-process Transformers + BitsAndBytes pairwise SceneFit judge."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from app.services.benchmark.pairwise_judge import (
    CRITERIA,
    PROMPT_VERSION,
    PairDecisionBatch,
    PairwiseGeminiJudge,
    RUBRIC,
)


TRANSFORMERS_JUDGE_BACKEND = "transformers-bnb-4bit"
DEFAULT_TRANSFORMERS_JUDGE_MODEL = "google/gemma-4-26B-A4B-it"
TRANSFORMERS_QUANTIZATION = "bitsandbytes-nf4-4bit"
DEFAULT_TRANSFORMERS_MAX_NEW_TOKENS = 768


class PairwiseTransformersJudge:
    """Run a Gemma vision-language judge in-process with 4-bit NF4 weights.

    The loaded model is deliberately protected by a lock. A single CUDA model
    instance is not safe to call concurrently, and serial generation also avoids
    multiplying its KV-cache allocation.
    """

    prompt_version = PROMPT_VERSION
    judge_backend = TRANSFORMERS_JUDGE_BACKEND
    quantization = TRANSFORMERS_QUANTIZATION

    def __init__(
        self,
        model_name: str = DEFAULT_TRANSFORMERS_JUDGE_MODEL,
        max_attempts: int = 3,
        max_new_tokens: int = DEFAULT_TRANSFORMERS_MAX_NEW_TOKENS,
        base_delay: float = 2.0,
        model: Any | None = None,
        processor: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.max_new_tokens = max_new_tokens
        self.base_delay = base_delay
        self.sleep = sleep
        self._generation_lock = threading.Lock()
        self._torch: Any | None = None
        if model is None or processor is None:
            self.model, self.processor, self._torch = self._load_model(model_name)
        else:
            self.model = model
            self.processor = processor

    @staticmethod
    def _load_model(model_name: str) -> tuple[Any, Any, Any]:
        """Load the official HF checkpoint with runtime NF4 quantization."""
        try:
            import torch
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers, accelerate, bitsandbytes, torch, and Pillow "
                "before using --judge-backend transformers-bnb-4bit"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("transformers-bnb-4bit judging requires a CUDA GPU")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model, processor, torch

    def artifact_metadata(self) -> dict[str, str]:
        return {
            "judge_backend": self.judge_backend,
            "judge_quantization": self.quantization,
        }

    @staticmethod
    def _image(path: Path) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install Pillow before local Transformers judging") from exc
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    @staticmethod
    def _json_instruction(pairs: Sequence[tuple[str, Path, Path]]) -> str:
        return PairwiseGeminiJudge._completion_instruction(pairs) + """

Return ONLY a valid JSON object, with no Markdown fence or surrounding prose.
Its exact top-level shape is:
{"pairs":[{"pair_id":"required ID","decisions":[
  {"criterion":"climate_season","winner":"left|right|tie","reason":"at most 20 words"},
  {"criterion":"activity_occasion","winner":"left|right|tie","reason":"at most 20 words"},
  {"criterion":"style_theme","winner":"left|right|tie","reason":"at most 20 words"},
  {"criterion":"color_harmony","winner":"left|right|tie","reason":"at most 20 words"},
  {"criterion":"overall","winner":"left|right|tie","reason":"at most 20 words"}
]}]}
"""

    def _messages(
        self, scene_path: Path, pairs: Sequence[tuple[str, Path, Path]]
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "SCENE:"},
            {"type": "image", "image": self._image(scene_path)},
        ]
        for pair_id, left_path, right_path in pairs:
            content.extend(
                (
                    {"type": "text", "text": f"PAIR ID: {pair_id}; LEFT OUTFIT:"},
                    {"type": "image", "image": self._image(left_path)},
                    {"type": "text", "text": "RIGHT OUTFIT:"},
                    {"type": "image", "image": self._image(right_path)},
                )
            )
        content.append({"type": "text", "text": self._json_instruction(pairs)})
        return [
            {"role": "system", "content": [{"type": "text", "text": RUBRIC}]},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _parse_json(text: str) -> PairDecisionBatch:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("Transformers judge returned no JSON object")
        try:
            payload, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError as exc:
            raise ValueError("Transformers judge returned invalid JSON") from exc
        return PairDecisionBatch.model_validate(payload)

    def _request(
        self, scene_path: Path, pairs: Sequence[tuple[str, Path, Path]]
    ) -> PairDecisionBatch:
        messages = self._messages(scene_path, pairs)
        with self._generation_lock:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            device = getattr(self.model, "device", None)
            if device is not None and hasattr(inputs, "to"):
                inputs = inputs.to(device)
            generation_context = (
                self._torch.inference_mode() if self._torch is not None else _NullContext()
            )
            with generation_context:
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                )
            prompt_length = inputs["input_ids"].shape[-1]
            text = self.processor.batch_decode(
                output_ids[:, prompt_length:], skip_special_tokens=True
            )[0]
        return self._parse_json(text)

    def compare_batch(
        self, scene_path: Path, pairs: Sequence[tuple[str, Path, Path]]
    ) -> list[dict[str, Any]]:
        expected_ids = [pair_id for pair_id, _, _ in pairs]
        pair_by_id = {pair_id: pair for pair_id, *pair in pairs}
        if len(pair_by_id) != len(expected_ids):
            raise ValueError("Pair request contains duplicate pair IDs")
        collected: dict[str, dict[str, Any]] = {}
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                pending_ids = [pair_id for pair_id in expected_ids if pair_id not in collected]
                pending_pairs = [(pair_id, *pair_by_id[pair_id]) for pair_id in pending_ids]
                parsed = self._request(scene_path, pending_pairs)
                received = {item.pair_id: item for item in parsed.pairs}
                if len(received) != len(parsed.pairs):
                    raise ValueError("Judge returned duplicate pair IDs")
                unexpected = set(received) - set(pending_ids)
                if unexpected:
                    raise ValueError(f"Judge returned unexpected pair IDs {sorted(unexpected)}")
                for pair_id, decision in received.items():
                    by_criterion = {item.criterion: item for item in decision.decisions}
                    if set(by_criterion) != set(CRITERIA) or len(by_criterion) != len(decision.decisions):
                        raise ValueError(
                            f"Pair {pair_id} did not return exactly the required criteria"
                        )
                    collected[pair_id] = {
                        "pair_id": pair_id,
                        "decisions": [by_criterion[criterion].model_dump() for criterion in CRITERIA],
                    }
                if len(collected) == len(expected_ids):
                    return [collected[pair_id] for pair_id in expected_ids]
                last_error = ValueError(
                    f"Judge omitted pair IDs {[pair_id for pair_id in expected_ids if pair_id not in collected]}"
                )
            except Exception as exc:
                last_error = exc
                print(
                    f"[PAIRWISE] Transformers attempt {attempt + 1}/{self.max_attempts} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            if attempt + 1 < self.max_attempts:
                self.sleep(self.base_delay * (2**attempt))
        missing = [pair_id for pair_id in expected_ids if pair_id not in collected]
        raise RuntimeError(
            f"Local Transformers pairwise judging failed with unresolved pair IDs {missing}"
        ) from last_error


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None
