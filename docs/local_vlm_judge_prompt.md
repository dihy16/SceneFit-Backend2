# External Local VLM Judge Prompt

Copy the prompt below to a vision-capable local agent. Give that agent this
repository, the extracted benchmark dataset, and the exact prepared manifest.
The agent must create a compatible `latest.zip` so that the Colab workflow can
run CLIP and Aesthetic retrieval without calling Gemini.

## Prompt to give the local agent

```text
You are the independent ground-truth VLM judge for the SceneFit retrieval
benchmark. Work in this repository. Your output will be used to evaluate
retrieval methods, so do not run CLIP, Aesthetic, VLM/SaMaG-R, ImageEdit, or
inspect any retrieval rankings.

Inputs you must use exactly:

1. The repository root contains extracted image data under data/bg and data/2d.
2. The exact candidate set is in results/benchmark/latest/manifest.json.
3. Do not regenerate, edit, reorder, or resample the manifest.

Score every pair in the manifest: 10 scenes x 100 outfits = exactly 1,000
scores. You must visually inspect the scene and outfit image for each pair.
If you cannot inspect an image, stop and report the blocker; do not fabricate a
score.

For each pair, assign one integer relevance score:

5 = excellent, highly suitable match
4 = good match with only minor issues
3 = acceptable or neutral match
2 = poor match with substantial incompatibility
1 = clearly unsuitable match

Judge compatibility independently. Consider climate/season, activity or
occasion, style/theme, and color harmony. Ignore image quality, pose, body
shape, and presentation. A scene can have many score-4 or score-5 outfits;
there is no single correct outfit.

Use temporary short aliases such as O01 through O100 internally if that makes
the visual review safer, but map each score back to the exact outfit ID in the
manifest. Do not use the aliases in the final artifacts.

Create results/benchmark/external-judge and copy the supplied manifest there
unchanged. Write progress incrementally so the task can resume. The final
results/benchmark/external-judge/judgments.jsonl must contain exactly 1,000
non-empty JSON lines: one unique (scene_id, outfit_id) pair for every manifest
combination. Every line must have this schema:

{
  "scene_id": "exact manifest scene ID",
  "outfit_id": "exact manifest outfit ID",
  "scene_sha256": "matching manifest scene sha256",
  "outfit_sha256": "matching manifest outfit sha256",
  "score": 1,
  "reason": "one concise visual compatibility reason",
  "judge_model": "local-agent-vlm-v1",
  "prompt_version": "scene-outfit-compatibility-v1"
}

Set score to an integer from 1 through 5. Never write duplicate pairs, unknown
IDs, missing hashes, floating-point scores, Markdown, or a JSON array in the
JSONL file.

Create results/benchmark/external-judge/judgments.meta.json using the exact
JSON object below. Compute manifest_fingerprint by importing and calling
app.services.benchmark.core.manifest_fingerprint on the copied manifest; do
not hand-calculate or substitute it.

{
  "version": 1,
  "manifest_fingerprint": "computed from copied manifest",
  "judge_model": "local-agent-vlm-v1",
  "prompt_version": "scene-outfit-compatibility-v1",
  "batch_size": 1
}

Create results/benchmark/external-judge/judge_info.json. Record the actual
local agent/model identity if known, the date, and state that
local-agent-vlm-v1 is the stable artifact identifier.

Create results/benchmark/external-judge/checkpoint.json using this structure,
substituting num_scenes, num_outfits, and seed from the manifest:

{
  "version": 2,
  "configuration": {
    "num_scenes": 10,
    "num_outfits": 100,
    "seed": 42,
    "model": "local-agent-vlm-v1",
    "batch_size": 1,
    "methods": ["clip", "aesthetic"]
  },
  "pilot_completed": true,
  "judged_scene_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "retrieved_scene_indices": []
}

Before packaging, run this command from the repository root. It must pass:

python scripts/benchmark.py validate-judgments \
  --manifest results/benchmark/external-judge/manifest.json \
  --judgments results/benchmark/external-judge/judgments.jsonl \
  --model local-agent-vlm-v1 \
  --batch-size 1

Package the CONTENTS of results/benchmark/external-judge into
results/benchmark/local-agent-latest.zip. The ZIP root must directly contain
manifest.json, judgments.jsonl, judgments.meta.json, checkpoint.json, and
judge_info.json; do not put them inside an external-judge/ folder. Report the
absolute path to local-agent-latest.zip, the pair count, the validation output,
and the score distribution.
```

## Importing the completed archive into Colab

Upload the agent's `local-agent-latest.zip` to Drive with this destination name:

```text
MyDrive/VRetrieval/benchmark/checkpoints-local-agent-v1/latest.zip
```

In `benchmark_colab.ipynb`, use this configuration before the prepare cell:

```python
CHECKPOINT_DIR = Path(
    '/content/drive/MyDrive/VRetrieval/benchmark/checkpoints-local-agent-v1'
)
JUDGE_MODEL = 'local-agent-vlm-v1'
BATCH_SIZE = 1
METHODS = ['clip', 'aesthetic']
```

If this Colab session already has an old generated run, preserve it before
restoring the agent archive:

```bash
cd /content/SceneFit-Backend2
mv results/benchmark/latest results/benchmark/latest-previous
```

Run the notebook's `prepare` cell to restore and validate the archive. Skip the
`judge` cell. Start Uvicorn in the Colab terminal, run the notebook readiness
cell, then run the `retrieve` cell. The resulting `final.zip` contains the
external VLM ground truth plus CLIP and Aesthetic rankings and metrics.
