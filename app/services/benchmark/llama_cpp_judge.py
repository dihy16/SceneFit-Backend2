"""Local llama.cpp-backed, blinded pairwise SceneFit judge."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx

from app.services.benchmark.pairwise_judge import (
    CRITERIA,
    PROMPT_VERSION,
    PairDecisionBatch,
    PairwiseGeminiJudge,
    RUBRIC,
)


LOCAL_JUDGE_BACKEND = "llama-cpp"
DEFAULT_LOCAL_JUDGE_MODEL = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0"
DEFAULT_LOCAL_JUDGE_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LOCAL_REQUEST_TIMEOUT = 300.0
LOCAL_QUANTIZATION = "Q4_0"


class PairwiseLlamaCppJudge:
    """Pairwise judge served locally by llama.cpp's OpenAI-compatible API."""

    prompt_version = PROMPT_VERSION
    judge_backend = LOCAL_JUDGE_BACKEND
    quantization = LOCAL_QUANTIZATION

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_JUDGE_MODEL,
        base_url: str = DEFAULT_LOCAL_JUDGE_BASE_URL,
        max_attempts: int = 3,
        request_timeout: float = DEFAULT_LOCAL_REQUEST_TIMEOUT,
        base_delay: float = 2.0,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_attempts = max_attempts
        self.request_timeout = request_timeout
        self.base_delay = base_delay
        self.sleep = sleep
        self.client = client or httpx.Client(timeout=request_timeout)

    def artifact_metadata(self) -> dict[str, str]:
        return {
            "judge_backend": self.judge_backend,
            "judge_quantization": self.quantization,
        }

    @staticmethod
    def _image_part(path: Path) -> dict[str, Any]:
        mime_type = PairwiseGeminiJudge._mime_type(path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        }

    def _messages(
        self,
        scene_path: Path,
        pairs: Sequence[tuple[str, Path, Path]],
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "SCENE:"},
            self._image_part(scene_path),
        ]
        for pair_id, left_path, right_path in pairs:
            content.extend(
                (
                    {"type": "text", "text": f"PAIR ID: {pair_id}; LEFT OUTFIT:"},
                    self._image_part(left_path),
                    {"type": "text", "text": "RIGHT OUTFIT:"},
                    self._image_part(right_path),
                )
            )
        content.append(
            {
                "type": "text",
                "text": PairwiseGeminiJudge._completion_instruction(pairs),
            }
        )
        return [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": content},
        ]

    def verify_server(self) -> None:
        response = self.client.get(f"{self.base_url}/models")
        response.raise_for_status()
        payload = response.json()
        available = {
            str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)
        }
        if available and self.model_name not in available:
            raise RuntimeError(
                f"llama.cpp does not serve requested model '{self.model_name}'; "
                f"available models: {sorted(available)}"
            )

    def _request(self, scene_path: Path, pairs: Sequence[tuple[str, Path, Path]]) -> PairDecisionBatch:
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model_name,
                "messages": self._messages(scene_path, pairs),
                "temperature": 0,
                "max_tokens": 768,
                "stream": False,
                "response_format": {
                    "type": "json_object",
                    "schema": PairDecisionBatch.model_json_schema(),
                },
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_effort": "none",
                "reasoning_format": "none",
            },
        )
        response.raise_for_status()
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("llama.cpp response did not contain chat completion text") from exc
        return PairDecisionBatch.model_validate_json(content)

    def compare_batch(
        self,
        scene_path: Path,
        pairs: Sequence[tuple[str, Path, Path]],
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
                pending_pairs = [
                    (pair_id, *pair_by_id[pair_id]) for pair_id in pending_ids
                ]
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
                        "decisions": [
                            by_criterion[criterion].model_dump() for criterion in CRITERIA
                        ],
                    }
                if len(collected) == len(expected_ids):
                    return [collected[pair_id] for pair_id in expected_ids]
                missing = [pair_id for pair_id in expected_ids if pair_id not in collected]
                last_error = ValueError(f"Judge omitted pair IDs {missing}")
            except Exception as exc:
                last_error = exc
                print(
                    f"[PAIRWISE] Local attempt {attempt + 1}/{self.max_attempts} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            if attempt + 1 < self.max_attempts:
                self.sleep(self.base_delay * (2**attempt))
        missing = [pair_id for pair_id in expected_ids if pair_id not in collected]
        raise RuntimeError(
            f"Local llama.cpp pairwise judging failed with unresolved pair IDs {missing}"
        ) from last_error
