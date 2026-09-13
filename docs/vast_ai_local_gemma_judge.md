# Run the Pairwise Judge Locally on Vast.ai

This replaces Gemini API judging with a local `llama.cpp` server running
Google's instruction-tuned Gemma 4 26B A4B QAT Q4_0 model. It produces a new,
separate ground-truth directory; do not mix it with the earlier API run.

## Instance and data

Use a CUDA Vast.ai instance with persistent disk. A 48 GB GPU is recommended
for four concurrent requests; use a 24 GB GPU only with one or two requests.
Restore the SceneFit repository, the dataset images, and the manifest just as
for the existing benchmark. The model download is about 15.6 GB.

The benchmark needs the source images referenced by:

```bash
results/benchmark/all-methods-evaluation/manifest.json
```

The previously collected retrieval rankings are already in:

```bash
results/benchmark/all-methods-evaluation/rankings
```

No `GEMINI_API_KEY` is needed. `HF_TOKEN` is only needed if Hugging Face asks
for authentication while downloading the model.

## Start llama.cpp

In terminal 1 on the Vast.ai instance, install the current CUDA build of
llama.cpp, then start the model server. The official model repository contains
both the Q4_0 GGUF and its required vision projector.

```bash
curl -LsSf https://llama.app/install.sh | sh

llama serve \
  -hf google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0 \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 16384 \
  --parallel 4 \
  --n-gpu-layers all
```

For a 24 GB GPU, replace `--ctx-size 16384 --parallel 4` with
`--ctx-size 8192 --parallel 2` and use benchmark concurrency 2 below.

Wait for the model to load, then check it from terminal 2:

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/models
```

Keep terminal 1 running for the entire judge job. The local server is bound to
localhost, so it is not exposed to the Internet.

## Pilot and judge

In terminal 2, activate the SceneFit environment. First run one pilot. It does
not save a benchmark label.

```bash
cd /workspace/SceneFit-Backend2
source .venv/bin/activate

python scripts/benchmark.py pilot \
  --judge-backend llama-cpp \
  --judge-base-url http://127.0.0.1:8080/v1 \
  --model google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0 \
  --manifest results/benchmark/all-methods-evaluation/manifest.json
```

If the pilot returns a valid five-criterion decision, start or resume the
clean local tournament:

```bash
python scripts/benchmark.py judge \
  --judge-backend llama-cpp \
  --judge-base-url http://127.0.0.1:8080/v1 \
  --model google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0 \
  --manifest results/benchmark/all-methods-evaluation/manifest.json \
  --judge-dir results/benchmark/pairwise_ground_truth_gemma4_26b_q4_local \
  --pairs-per-request 1 \
  --concurrency 4 \
  --max-attempts 3
```

For the 24 GB profile, set `--concurrency 2`. `Ctrl+C` is safe: every saved
response and reconciled comparison is checkpointed, and rerunning the same
command resumes the job.

## Validate and score the existing rankings

After the judge completes, validate the clean local ground truth:

```bash
python scripts/benchmark.py validate-judge \
  --judge-backend llama-cpp \
  --manifest results/benchmark/all-methods-evaluation/manifest.json \
  --judge-dir results/benchmark/pairwise_ground_truth_gemma4_26b_q4_local \
  --model google/gemma-4-26B-A4B-it-qat-q4_0-gguf:Q4_0
```

Then evaluate the already saved `clip`, `aesthetic`, `vlm`, and `image_edit`
rankings. This requires neither the server nor a GPU:

```bash
python scripts/benchmark.py evaluate \
  --manifest results/benchmark/all-methods-evaluation/manifest.json \
  --rankings-dir results/benchmark/all-methods-evaluation/rankings \
  --judge-dir results/benchmark/pairwise_ground_truth_gemma4_26b_q4_local \
  --output-dir results/benchmark/pairwise_evaluation_gemma4_26b_q4_local \
  --protocol pairwise-elo-v1
```

Download the two output directories before stopping the Vast.ai instance.
