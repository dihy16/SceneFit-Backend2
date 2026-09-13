"""Unit tests for the benchmark; no GPU, API key, or live worker is required."""

from __future__ import annotations

import json
import json as json_module
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from app.services.benchmark.collector import collect_rankings, normalize_ranking
from app.services.benchmark.core import (
    create_manifest,
    run_judging,
    validate_judgments,
    write_json,
)
from app.services.benchmark.gemini_judge import (
    DEFAULT_JUDGE_MODEL,
    GeminiJudge,
    JudgmentBatch,
    OutfitJudgment,
)
from app.services.benchmark.metrics import (
    evaluate_benchmark,
    evaluate_pairwise_benchmark,
    ndcg,
)
from app.services.benchmark.pairwise import (
    DEFAULT_CONCURRENCY,
    DEFAULT_PAIRS_PER_REQUEST,
    run_pairwise_judging,
    validate_pairwise_judge,
)
from app.services.benchmark.pairwise_judge import (
    CRITERIA,
    PROMPT_VERSION,
    RUBRIC,
    PairwiseGeminiJudge,
)
from app.services.benchmark.llama_cpp_judge import (
    DEFAULT_LOCAL_JUDGE_MODEL,
    PairwiseLlamaCppJudge,
)
from scripts.benchmark import build_parser as build_benchmark_parser
from scripts.run_benchmark_colab import build_parser as build_colab_parser
from scripts.run_benchmark_runtime import (
    _load_state,
    _restore,
    _save_checkpoint,
    _validate_state,
    build_parser as build_runtime_parser,
)
from scripts.build_pe_index import _image_paths


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


class FakePairwiseJudge:
    model_name = "fake-pairwise"
    prompt_version = PROMPT_VERSION

    def compare_batch(self, scene_path, pairs):
        return [
            {
                "pair_id": pair_id,
                "decisions": [
                    {"criterion": criterion, "winner": "left", "reason": "fixture"}
                    for criterion in CRITERIA
                ],
            }
            for pair_id, _, _ in pairs
        ]


class ConsistentFakePairwiseJudge(FakePairwiseJudge):
    def compare_batch(self, scene_path, pairs):
        return [
            {
                "pair_id": pair_id,
                "decisions": [
                    {
                        "criterion": criterion,
                        "winner": "left" if left.name < right.name else "right",
                        "reason": "fixture",
                    }
                    for criterion in CRITERIA
                ],
            }
            for pair_id, left, right in pairs
        ]


class FakePartialPairwiseModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        pair_id = "p1" if self.calls == 1 else "p2"
        response = type("Response", (), {})()
        response.parsed = {
            "pairs": [
                {
                    "pair_id": pair_id,
                    "decisions": [
                        {
                            "criterion": criterion,
                            "winner": "left",
                            "reason": "fixture",
                        }
                        for criterion in CRITERIA
                    ],
                }
            ]
        }
        response.text = ""
        return response


class FakePartialPairwiseClient:
    def __init__(self):
        self.models = FakePartialPairwiseModels()


class FakeModels:
    def __init__(self):
        self.calls = 0
        self.last_request = None

    def generate_content(self, **kwargs):
        self.calls += 1
        self.last_request = kwargs
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


class FakeMalformedModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        response = type("Response", (), {})()
        if self.calls == 1:
            judgments = [
                OutfitJudgment(outfit_id="o1", score=5, reason="first"),
                OutfitJudgment(outfit_id="o1", score=4, reason="duplicate"),
            ]
        else:
            judgments = [
                OutfitJudgment(outfit_id="o1", score=5, reason="first"),
                OutfitJudgment(outfit_id="o2", score=4, reason="second"),
            ]
        response.parsed = JudgmentBatch(judgments=judgments)
        response.text = ""
        return response


class FakeMalformedClient:
    def __init__(self):
        self.models = FakeMalformedModels()


class FakeExtraDuplicateModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        response = type("Response", (), {})()
        response.parsed = JudgmentBatch(
            judgments=[
                OutfitJudgment(outfit_id="o1", score=5, reason="first"),
                OutfitJudgment(outfit_id="o2", score=4, reason="second"),
                OutfitJudgment(outfit_id="o2", score=3, reason="duplicate"),
            ]
        )
        response.text = ""
        return response


class FakeExtraDuplicateClient:
    def __init__(self):
        self.models = FakeExtraDuplicateModels()


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeLlamaResponse(FakeResponse):
    pass


class FakeLlamaClient:
    def __init__(self):
        self.requests = []

    def get(self, url):
        return FakeLlamaResponse({"data": [{"id": DEFAULT_LOCAL_JUDGE_MODEL}]})

    def post(self, url, json):
        self.requests.append((url, json))
        pair_id = json["messages"][1]["content"][2]["text"].split(";", 1)[0].split(": ", 1)[1]
        content = {
            "pairs": [
                {
                    "pair_id": pair_id,
                    "decisions": [
                        {"criterion": criterion, "winner": "left", "reason": "fixture"}
                        for criterion in CRITERIA
                    ],
                }
            ]
        }
        return FakeLlamaResponse({"choices": [{"message": {"content": json_module.dumps(content)}}]})


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

    def test_complete_judgments_are_required_before_retrieval(self):
        manifest = self.manifest()
        output = self.root / "judgments.jsonl"
        judge = FakeJudge()
        first_scene = manifest["scenes"][0]["id"]
        run_judging(
            manifest,
            self.root,
            output,
            judge,
            batch_size=3,
            scene_ids=[first_scene],
        )
        with self.assertRaisesRegex(ValueError, "3/6 pairs"):
            validate_judgments(
                manifest, output, judge.model_name, judge.prompt_version, 3
            )
        run_judging(manifest, self.root, output, judge, batch_size=3)
        self.assertEqual(
            validate_judgments(
                manifest, output, judge.model_name, judge.prompt_version, 3
            ),
            6,
        )

    def test_duplicate_judgments_are_rejected(self):
        manifest = self.manifest()
        output = self.root / "judgments.jsonl"
        judge = FakeJudge()
        run_judging(manifest, self.root, output, judge, batch_size=3)
        first = output.read_text(encoding="utf-8").splitlines()[0]
        with output.open("a", encoding="utf-8") as stream:
            stream.write(first + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate judgment"):
            validate_judgments(
                manifest, output, judge.model_name, judge.prompt_version, 3
            )

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

    def test_pairwise_judge_is_mirrored_resumable_and_validated(self):
        (self.outfits / "outfit_3.png").write_bytes(b"outfit-3")
        manifest = create_manifest(self.root, self.scenes, self.outfits, 2, 4, seed=42)
        output = self.root / "pairwise"
        judge = FakePairwiseJudge()
        self.assertEqual(
            run_pairwise_judging(
                manifest,
                self.root,
                output,
                judge,
                rounds=2,
                pairs_per_request=1,
                concurrency=2,
            ),
            8,
        )
        self.assertEqual(
            validate_pairwise_judge(
                manifest, output, judge.model_name, rounds=2, pairs_per_request=1
            ),
            8,
        )
        self.assertEqual(
            run_pairwise_judging(
                manifest, self.root, output, judge, rounds=2, pairs_per_request=1
            ),
            0,
        )
        responses = [
            json.loads(line)
            for line in (output / "judge_responses.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(responses), 16)
        progress = json.loads((output / "progress.json").read_text())
        self.assertEqual(progress["status"], "complete")
        self.assertEqual(progress["completed_comparisons"], 8)
        ratings = json.loads((output / "ratings.json").read_text())
        overall = ratings["scenes"][manifest["scenes"][0]["id"]]["criteria"]["overall"]
        self.assertEqual(len(overall), 4)

    def test_pairwise_prompt_uses_balanced_few_shots_and_final_id_checklist(self):
        judge = PairwiseGeminiJudge(client=FakeClient())
        pairs = [
            (
                "scene:r1:p001",
                self.outfits / "outfit_0.png",
                self.outfits / "outfit_1.png",
            ),
            (
                "scene:r1:p002",
                self.outfits / "outfit_1.png",
                self.outfits / "outfit_2.png",
            ),
        ]
        contents = judge._contents(self.scenes / "scene_0.png", pairs)
        final_instruction = contents[-1]

        self.assertIn("RIGHT should win overall", RUBRIC)
        self.assertIn("LEFT should win overall", RUBRIC)
        self.assertIn("genuine tie", RUBRIC)
        self.assertIn("Return exactly 2 pair objects", final_instruction)
        self.assertIn("1. scene:r1:p001", final_instruction)
        self.assertIn("2. scene:r1:p002", final_instruction)
        self.assertTrue(PROMPT_VERSION.endswith("v2-fewshot"))

    def test_pairwise_judge_repairs_only_omitted_pair_ids(self):
        client = FakePartialPairwiseClient()
        judge = PairwiseGeminiJudge(
            client=client,
            base_delay=0,
            sleep=lambda _: None,
        )
        pairs = [
            ("p1", self.outfits / "outfit_0.png", self.outfits / "outfit_1.png"),
            ("p2", self.outfits / "outfit_1.png", self.outfits / "outfit_2.png"),
        ]

        result = judge.compare_batch(self.scenes / "scene_0.png", pairs)

        self.assertEqual([item["pair_id"] for item in result], ["p1", "p2"])
        self.assertEqual(client.models.calls, 2)

    def test_local_llama_cpp_pairwise_judge_uses_images_and_json_schema(self):
        client = FakeLlamaClient()
        judge = PairwiseLlamaCppJudge(client=client, base_delay=0, sleep=lambda _: None)
        judge.verify_server()
        result = judge.compare_batch(
            self.scenes / "scene_0.png",
            [("p1", self.outfits / "outfit_0.png", self.outfits / "outfit_1.png")],
        )

        self.assertEqual(result[0]["pair_id"], "p1")
        self.assertEqual(result[0]["decisions"][0]["criterion"], CRITERIA[0])
        url, request = client.requests[0]
        self.assertTrue(url.endswith("/v1/chat/completions"))
        self.assertEqual(request["model"], DEFAULT_LOCAL_JUDGE_MODEL)
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertIn("schema", request["response_format"])
        image_parts = [
            part for part in request["messages"][1]["content"]
            if part["type"] == "image_url"
        ]
        self.assertEqual(len(image_parts), 3)
        self.assertTrue(image_parts[0]["image_url"]["url"].startswith("data:image/"))
        self.assertEqual(judge.artifact_metadata()["judge_backend"], "llama-cpp")

    def test_pairwise_validation_rejects_wrong_backend(self):
        (self.outfits / "outfit_3.png").write_bytes(b"outfit-3")
        manifest = create_manifest(self.root, self.scenes, self.outfits, 2, 4, seed=42)
        output = self.root / "local-pairwise"

        class LocalFixtureJudge(FakePairwiseJudge):
            def artifact_metadata(self):
                return {"judge_backend": "llama-cpp", "judge_quantization": "Q4_0"}

        judge = LocalFixtureJudge()
        run_pairwise_judging(manifest, self.root, output, judge, rounds=1)
        self.assertEqual(
            validate_pairwise_judge(
                manifest, output, judge.model_name, rounds=1, judge_backend="llama-cpp"
            ),
            4,
        )
        with self.assertRaisesRegex(ValueError, "judge_backend"):
            validate_pairwise_judge(
                manifest, output, judge.model_name, rounds=1, judge_backend="gemini"
            )

    def test_pairwise_judging_resumes_a_partially_reconciled_round(self):
        (self.outfits / "outfit_3.png").write_bytes(b"outfit-3")
        manifest = create_manifest(self.root, self.scenes, self.outfits, 2, 4, seed=42)
        output = self.root / "partial-pairwise"
        scene_ids = [manifest["scenes"][0]["id"]]
        run_pairwise_judging(
            manifest,
            self.root,
            output,
            FakePairwiseJudge(),
            rounds=2,
            pairs_per_request=1,
            scene_ids=scene_ids,
        )
        comparison_path = output / "comparisons.jsonl"
        saved = comparison_path.read_text().splitlines()
        comparison_path.write_text(saved[0] + "\n", encoding="utf-8")

        written = run_pairwise_judging(
            manifest,
            self.root,
            output,
            FakePairwiseJudge(),
            rounds=2,
            pairs_per_request=1,
            scene_ids=scene_ids,
        )

        self.assertEqual(written, 3)
        self.assertEqual((output / "comparisons.jsonl").read_text().count("\n"), 4)

    def test_pairwise_metrics_reward_an_elo_ordered_ranking(self):
        (self.outfits / "outfit_3.png").write_bytes(b"outfit-3")
        manifest = create_manifest(self.root, self.scenes, self.outfits, 2, 4, seed=42)
        judge_dir = self.root / "pairwise"
        judge = ConsistentFakePairwiseJudge()
        run_pairwise_judging(manifest, self.root, judge_dir, judge, rounds=2)
        rankings_dir = self.root / "rankings"
        ratings = json.loads((judge_dir / "ratings.json").read_text())
        scenes = {}
        for scene in manifest["scenes"]:
            ordered = ratings["scenes"][scene["id"]]["criteria"]["overall"]
            scenes[scene["id"]] = [
                {"outfit_id": item["outfit_id"], "rank": item["rank"], "raw_score": None}
                for item in ordered
            ]
        write_json(rankings_dir / "ideal.json", {"version": 1, "method": "ideal", "scenes": scenes})
        result = evaluate_pairwise_benchmark(
            manifest,
            judge_dir / "comparisons.jsonl",
            judge_dir / "ratings.json",
            rankings_dir,
            self.root / "output",
        )
        self.assertAlmostEqual(result["metrics"]["ideal"]["pairwise_agreement"], 1.0)
        self.assertAlmostEqual(result["metrics"]["ideal"]["kendall_tau"], 1.0)

    def test_gemini_retries(self):
        client = FakeClient()
        judge = GeminiJudge(client=client, base_delay=0, sleep=lambda _: None)
        result = judge.score_batch(self.scenes / "scene_0.png", [("o1", self.outfits / "outfit_0.png")])
        self.assertEqual(result[0]["score"], 5)
        self.assertEqual(client.models.calls, 2)
        self.assertEqual(client.models.last_request["model"], DEFAULT_JUDGE_MODEL)

    def test_gemini_retries_malformed_outfit_ids(self):
        client = FakeMalformedClient()
        judge = GeminiJudge(client=client, base_delay=0, sleep=lambda _: None)
        result = judge.score_batch(
            self.scenes / "scene_0.png",
            [
                ("o1", self.outfits / "outfit_0.png"),
                ("o2", self.outfits / "outfit_1.png"),
            ],
        )
        self.assertEqual([item["outfit_id"] for item in result], ["o1", "o2"])
        self.assertEqual(client.models.calls, 2)

    def test_gemini_discards_extra_duplicate_when_all_ids_are_present(self):
        client = FakeExtraDuplicateClient()
        judge = GeminiJudge(client=client, base_delay=0, sleep=lambda _: None)
        result = judge.score_batch(
            self.scenes / "scene_0.png",
            [
                ("o1", self.outfits / "outfit_0.png"),
                ("o2", self.outfits / "outfit_1.png"),
            ],
        )
        self.assertEqual([item["outfit_id"] for item in result], ["o1", "o2"])
        self.assertEqual(result[1]["score"], 4)
        self.assertEqual(client.models.calls, 1)

    def test_cli_defaults_and_separate_stages(self):
        benchmark_parser = build_benchmark_parser()
        self.assertEqual(
            benchmark_parser.parse_args(["judge"]).model,
            DEFAULT_JUDGE_MODEL,
        )
        judge_defaults = benchmark_parser.parse_args(["judge"])
        self.assertEqual(judge_defaults.pairs_per_request, DEFAULT_PAIRS_PER_REQUEST)
        self.assertEqual(judge_defaults.concurrency, DEFAULT_CONCURRENCY)
        self.assertEqual(
            benchmark_parser.parse_args(["validate-judgments"]).model,
            DEFAULT_JUDGE_MODEL,
        )
        self.assertEqual(
            benchmark_parser.parse_args(["judge", "--scene-index", "2"]).scene_index,
            2,
        )
        colab_parser = build_colab_parser()
        self.assertTrue(colab_parser.parse_args(["--judge-only"]).judge_only)
        self.assertTrue(colab_parser.parse_args(["--retrieve-only"]).retrieve_only)
        self.assertEqual(colab_parser.parse_args([]).model, DEFAULT_JUDGE_MODEL)

    def test_runtime_checkpoint_round_trip_and_configuration_checks(self):
        runtime_parser = build_runtime_parser()
        checkpoint_dir = self.root / "checkpoints"
        parsed = runtime_parser.parse_args(
            ["prepare", "--checkpoint-dir", str(checkpoint_dir)]
        )
        self.assertEqual(parsed.methods, ["clip", "aesthetic"])
        self.assertEqual(parsed.model, DEFAULT_JUDGE_MODEL)

        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
        state = {
            "version": 3,
            "configuration": {"model": DEFAULT_JUDGE_MODEL},
            "pilot_completed": True,
            "judged_scene_indices": [0],
            "retrieved_scene_indices": [],
        }
        _save_checkpoint(run_dir, checkpoint_dir, "prepared", state)
        self.assertEqual(_load_state(checkpoint_dir / "latest.zip"), state)

        shutil.rmtree(run_dir)
        _restore(checkpoint_dir, run_dir)
        self.assertTrue((run_dir / "manifest.json").is_file())
        with self.assertRaisesRegex(ValueError, "different benchmark configuration"):
            _validate_state(state, {"model": "other-model"})

    def test_colab_notebooks_do_not_embed_ngrok_credentials(self):
        root = Path(__file__).parents[1]
        benchmark_notebook = json.loads(
            (root / "benchmark_colab.ipynb").read_text(encoding="utf-8")
        )
        old_notebook = (root / "run_colab.ipynb").read_text(encoding="utf-8")
        self.assertEqual(benchmark_notebook["nbformat"], 4)
        self.assertIn("userdata.get('NGROK_TOKEN')", old_notebook)
        self.assertNotIn("ngrok.set_auth_token(\"", old_notebook)

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
