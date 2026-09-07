"""Unit tests for the benchmark; no GPU, API key, or live worker is required."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from app.services.benchmark.collector import collect_rankings, normalize_ranking
from app.services.benchmark.core import (
    create_manifest,
    manifest_fingerprint,
    run_judging,
    write_json,
)
from app.services.benchmark.gemini_judge import GeminiJudge, JudgmentBatch, OutfitJudgment
from app.services.benchmark.metrics import evaluate_benchmark, ndcg
from scripts.build_pe_index import _image_paths
from scripts.run_benchmark_phase import _verify_complete_judgments


class FakeJudge:
    model_name = "fake-judge"
    prompt_version = "test-v1"

    def __init__(self):
        self.calls = 0

    def score_batch(self, scene_path, outfits):
        self.calls += 1
        return [
            {"outfit_id": outfit_id, "score": index + 1, "reason": "fixture"}
            for index, (outfit_id, _) in enumerate(outfits)
        ]


class FakeModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("rate limited")
        response = type("Response", (), {})()
        response.parsed = JudgmentBatch(
            judgments=[OutfitJudgment(outfit_id="o1", score=5, reason="excellent")]
        )
        response.text = ""
        return response


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.scenes = self.root / "scenes"
        self.outfits = self.root / "outfits"
        self.scenes.mkdir()
        self.outfits.mkdir()
        for index in range(2):
            (self.scenes / f"scene_{index}.png").write_bytes(f"scene-{index}".encode())
        for index in range(3):
            (self.outfits / f"outfit_{index}.png").write_bytes(f"outfit-{index}".encode())

    def tearDown(self):
        self.tempdir.cleanup()

    def manifest(self):
        return create_manifest(self.root, self.scenes, self.outfits, 2, 3, seed=42)

    def test_manifest_is_reproducible_and_rejects_insufficient_data(self):
        first = self.manifest()
        second = self.manifest()
        self.assertEqual(first["scenes"], second["scenes"])
        self.assertEqual(first["outfits"], second["outfits"])
        with self.assertRaisesRegex(ValueError, "only 2"):
            create_manifest(self.root, self.scenes, self.outfits, 3, 3)

    def test_pe_index_uses_only_manifest_outfits(self):
        manifest = self.manifest()
        manifest_path = self.root / "manifest.json"
        write_json(manifest_path, manifest)
        expected = [self.root / entry["path"] for entry in manifest["outfits"]]

        current_directory = Path.cwd()
        try:
            os.chdir(self.root)
            selected = _image_paths(self.outfits, manifest_path)
        finally:
            os.chdir(current_directory)

        self.assertEqual([path.resolve() for path in selected], expected)

    def test_heavy_phase_requires_complete_shared_judgments(self):
        manifest = self.manifest()
        judgments = self.root / "judgments.jsonl"
        rows = [
            {
                "scene_id": scene["id"],
                "outfit_id": outfit["id"],
                "score": 3,
            }
            for scene in manifest["scenes"]
            for outfit in manifest["outfits"]
        ]
        judgments.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        _verify_complete_judgments(manifest, judgments)

        judgments.write_text(
            "".join(json.dumps(row) + "\n" for row in rows[:-1]),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "missing=1"):
            _verify_complete_judgments(manifest, judgments)

    def test_judging_is_incremental_and_resumable(self):
        manifest = self.manifest()
        output = self.root / "judgments.jsonl"
        judge = FakeJudge()
        first_scene = manifest["scenes"][0]["id"]
        self.assertEqual(
            run_judging(
                manifest,
                self.root,
                output,
                judge,
                batch_size=2,
                scene_ids=[first_scene],
            ),
            3,
        )
        self.assertEqual(
            run_judging(
                manifest,
                self.root,
                output,
                judge,
                batch_size=2,
                scene_ids=[first_scene],
            ),
            0,
        )
        self.assertEqual(run_judging(manifest, self.root, output, judge, batch_size=2), 3)
        self.assertEqual(run_judging(manifest, self.root, output, judge, batch_size=2), 0)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(1 <= row["score"] <= 5 for row in rows))

    def test_normalize_ranking_requires_exact_pool(self):
        payload = [{"name": "a.png", "score": 0.9}, {"outfit_name": "b", "similarity": 0.8}]
        ranking = normalize_ranking(payload, ["a", "b"])
        self.assertEqual([item["outfit_id"] for item in ranking], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "missing"):
            normalize_ranking(payload[:1], ["a", "b"])

    def test_live_collection_is_cached_and_strict(self):
        manifest = self.manifest()
        config = self.root / "workers.yaml"
        config.write_text(
            "retrieval_methods:\n  clip:\n    url: https://worker.test\n"
            "    endpoint: api/retrieve\nretry:\n  max_attempts: 1\n  delay_seconds: 0\n",
            encoding="utf-8",
        )
        expected = [item["id"] for item in manifest["outfits"]]
        calls = []

        def request(url, **kwargs):
            calls.append((url, kwargs["data"]))
            return FakeResponse([{"name": item_id, "score": 1.0} for item_id in expected])

        output = self.root / "rankings"
        first_scene = manifest["scenes"][0]["id"]
        collect_rankings(
            manifest,
            self.root,
            config,
            output,
            request=request,
            scene_ids=[first_scene],
        )
        self.assertEqual(len(calls), 1)
        collect_rankings(manifest, self.root, config, output, request=request)
        collect_rankings(manifest, self.root, config, output, request=request)
        self.assertEqual(len(calls), 2)  # one request per scene; second run is cached
        self.assertEqual(json.loads(calls[0][1]["candidate_names"]), expected)

    def test_image_edit_collection_uses_precomputed_scene_prompt(self):
        manifest = self.manifest()
        config = self.root / "workers.yaml"
        config.write_text(
            "retrieval_methods:\n  image_edit:\n    url: http://127.0.0.1:8000\n"
            "    endpoint: api/v1/workers/image-edit-flux\n"
            "retry:\n  max_attempts: 1\n  delay_seconds: 0\n",
            encoding="utf-8",
        )
        run_dir = self.root / "run"
        rankings_dir = run_dir / "rankings"
        run_dir.mkdir()
        scene_id = manifest["scenes"][0]["id"]
        write_json(
            run_dir / "image_edit_prompts.json",
            {
                "version": 1,
                "manifest_fingerprint": manifest_fingerprint(manifest),
                "scenes": {scene_id: "A lightweight linen outfit"},
            },
        )
        expected = [item["id"] for item in manifest["outfits"]]
        requests = []

        def request(url, **kwargs):
            requests.append(kwargs["data"])
            return FakeResponse(
                [{"name": item_id, "score": 1.0} for item_id in expected]
            )

        collect_rankings(
            manifest,
            self.root,
            config,
            rankings_dir,
            methods=["image_edit"],
            request=request,
            scene_ids=[scene_id],
        )

        self.assertEqual(
            requests[0]["outfit_description"],
            "A lightweight linen outfit",
        )

    def test_live_worker_config_does_not_target_proxy_routes(self):
        config_path = Path(__file__).parents[1] / "config" / "retrieval_methods.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected_endpoints = {
            "clip": "api/v1/workers/clip",
            "aesthetic": "api/v1/workers/aesthetic",
            "vlm": "api/v1/workers/vlm-faiss-composed-retrieval",
            "image_edit": "api/v1/workers/image-edit-flux",
        }
        actual = {
            method: settings["endpoint"]
            for method, settings in config["retrieval_methods"].items()
        }
        self.assertEqual(actual, expected_endpoints)
        self.assertTrue(all("/retrieval/" not in endpoint for endpoint in actual.values()))

    def test_image_edit_package_import_targets_service_module(self):
        init_path = (
            Path(__file__).parents[1]
            / "app"
            / "services"
            / "image_edit"
            / "__init__.py"
        )
        source = init_path.read_text(encoding="utf-8")
        self.assertIn("from app.services.image_edit.service import", source)
        self.assertNotIn("app.services.image_edit.service.service", source)

    def test_ndcg_ideal_and_reversed(self):
        labels = [5, 4, 2, 1]
        self.assertAlmostEqual(ndcg(labels, labels, 4), 1.0)
        self.assertLess(ndcg(list(reversed(labels)), labels, 4), 1.0)

    def test_gemini_retries(self):
        client = FakeClient()
        judge = GeminiJudge(client=client, base_delay=0, sleep=lambda _: None)
        result = judge.score_batch(self.scenes / "scene_0.png", [("o1", self.outfits / "outfit_0.png")])
        self.assertEqual(result[0]["score"], 5)
        self.assertEqual(client.models.calls, 2)

    def test_small_end_to_end_evaluation(self):
        manifest = self.manifest()
        judgments = self.root / "judgments.jsonl"
        run_judging(manifest, self.root, judgments, FakeJudge(), batch_size=3)
        matrix = {}
        for line in judgments.read_text().splitlines():
            row = json.loads(line)
            matrix.setdefault(row["scene_id"], {})[row["outfit_id"]] = row["score"]

        rankings_dir = self.root / "rankings"
        rankings_dir.mkdir()
        scenes = {}
        for scene in manifest["scenes"]:
            ordered = sorted(matrix[scene["id"]], key=matrix[scene["id"]].get, reverse=True)
            scenes[scene["id"]] = [
                {"outfit_id": outfit_id, "rank": rank, "raw_score": 1 / rank}
                for rank, outfit_id in enumerate(ordered, 1)
            ]
        write_json(rankings_dir / "ideal.json", {"version": 1, "method": "ideal", "scenes": scenes})
        result = evaluate_benchmark(manifest, judgments, rankings_dir, self.root / "output", ks=(2, 3))
        self.assertAlmostEqual(result["metrics"]["ideal"]["ndcg@3"], 1.0)
        self.assertTrue((self.root / "output" / "per_scene.csv").exists())


if __name__ == "__main__":
    unittest.main()
