# Gemini Retrieval Benchmark

This benchmark treats each scene as a retrieval query and uses one shared outfit candidate pool. Gemini 3.7 Flash assigns a graded relevance score from 1 (unsuitable) to 5 (excellent) to every scene–outfit pair. Retrieval workers then rank the exact same candidates; the evaluator reports nDCG@5/10 and mean relevance@5/10.

## Setup

Add `GEMINI_API_KEY` to `.env` or the process environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Generated artifacts are written under `results/benchmark/` and are resumable. Free-tier Gemini inputs may be used by Google to improve its products; use a paid project if that is unsuitable for the dataset.

## Commands

First, run the offline smoke test. It uses two scenes, five outfits, a deterministic fixture judge, and synthetic rankings, so it requires no GPU, API key, or worker:

```bash
python scripts/benchmark.py smoke
```

Create the default 100-outfit, 10-scene manifest:

```bash
python scripts/benchmark.py manifest
```

For a smaller development run, explicitly choose the scene count:

```bash
python scripts/benchmark.py manifest --num-scenes 2
python scripts/benchmark.py judge
python scripts/benchmark.py collect
python scripts/benchmark.py evaluate
```

`judge` sends ten outfits per request by default. Use `--batch-size 1` for independent pairwise requests. `collect` calls worker URLs from `config/retrieval_methods.yaml` directly and requires each worker to support the multipart `candidate_names` JSON field and return every manifest outfit exactly once.

Run the complete resumable workflow with:

```bash
python scripts/benchmark.py run --num-scenes 10 --num-outfits 100
```

Set every selected worker URL in `config/retrieval_methods.yaml` before using `run`. The combined command collects worker rankings before spending Gemini quota, and each stage can also be resumed separately.

The output contains the data manifest and hashes, append-only `judgments.jsonl`, its evaluator metadata, cached rankings, `per_scene.csv`, and `summary.json`. Do not edit a manifest after judging begins; create a new run directory for a different sample.

## Running with a Google Colab GPU

Gemini 3.7 Flash runs in Google's API and does not use the Colab GPU. The GPU is useful for the CLIP, image-edit, VLM/SaMaG-R, and aesthetic retrieval workers. The benchmark CLI may run in the same Colab session or locally while calling remote GPU workers.

If all retrieval workers already run on other machines, the benchmark session itself does not benefit from a GPU; create a CPU session instead. The A100 command below is appropriate when this Colab environment also hosts retrieval-model work.

The official Colab CLI supports Linux and macOS. On Windows, run these commands from WSL, not PowerShell. Install and authenticate once:

```bash
uv tool install google-colab-cli
colab auth login
```

The provided setup script clones the `main` branch from GitHub. Commit and push
the benchmark implementation before using it, or the Colab VM will receive the
older repository version. Upload the full dataset archive to
`/MyDrive/VRetrieval/data.zip` in Google Drive. Setup clears the VM's previous
scenes, outfits, and active benchmark run, extracts `data.zip`, and defers index
construction so benchmark-mode startup can encode only the selected 100
outfits.

From the repository directory in WSL, create an A100 session and prepare it:

```bash
SESSION_NAME=scenefit-benchmark
colab new -s "$SESSION_NAME" --gpu A100
colab drivemount -s "$SESSION_NAME"
colab exec -s "$SESSION_NAME" --timeout 3600 -f setup_colab.py
colab upload -s "$SESSION_NAME" .env /content/.env
colab status -s "$SESSION_NAME"
```

Your `.env` must contain `GEMINI_API_KEY`. The worker URL configuration is
covered in the next section.

The Drive archive currently supplies six backgrounds. Upload the four local
`CamView` images individually after setup; keeping each transfer small is more
reliable than uploading one large scene archive:

```bash
colab upload -s "$SESSION_NAME" data/bg/CamView_01_Default.png /content/data/bg/CamView_01_Default.png
colab upload -s "$SESSION_NAME" data/bg/CamView_Capture_02.png /content/data/bg/CamView_Capture_02.png
colab upload -s "$SESSION_NAME" data/bg/CamView_Capture_03.png /content/data/bg/CamView_Capture_03.png
colab upload -s "$SESSION_NAME" data/bg/CamView_Current.png /content/data/bg/CamView_Current.png
```

Open a shell on the GPU VM:

```bash
colab console -s "$SESSION_NAME"
```

Inside the Colab shell, verify the GPU and run the offline smoke test first:

```bash
cd /content
nvidia-smi
test "$(find data/bg -maxdepth 1 -type f | wc -l)" -eq 10
test -f data/ref_images/man.png
test -f data/ref_images/woman.png
python scripts/benchmark.py smoke
exit
```

The two reference-image checks are required for the `image_edit` method. If a
check fails, add the missing file to `data.zip` or upload it before continuing.

## Start Retrieval Workers and Configure Their URLs

The benchmark evaluates retrieval *workers*, not just Gemini. Prepare the
manifest before starting the server so its vector database indexes only the
100 benchmark outfits instead of every image in `data/2d`. This command also
builds the separate PE-CLIP FAISS index required by the VLM method from those
same 100 outfits:

```bash
python scripts/run_benchmark_colab.py \
  --session "$SESSION_NAME" \
  --num-scenes 10 \
  --num-outfits 100 \
  --prepare-only
```

Start the FastAPI server afterward and keep it running for the full benchmark.
Open a second WSL terminal for this server process:

```bash
SESSION_NAME=scenefit-benchmark
colab console -s "$SESSION_NAME"
```

Inside that Colab console, run:

```bash
cd /content
BENCHMARK_MANIFEST=/content/results/benchmark/latest/manifest.json \
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The first benchmark-mode startup should show `Encoding outfits: ... 100/100`.
Later startups reuse `results/benchmark/latest/vector.index`. Wait until
Uvicorn reports that the application is running, and do not close this console.
Starting Uvicorn without `BENCHMARK_MANIFEST` retains the normal full-dataset
indexing behavior.

The benchmark driver runs on the same Colab VM, so the simplest and recommended
configuration uses its loopback address rather than Ngrok:

```yaml
retrieval_methods:
  clip:
    url: "http://127.0.0.1:8000"
    endpoint: "api/v1/workers/clip"
  image_edit:
    url: "http://127.0.0.1:8000"
    endpoint: "api/v1/workers/image-edit-flux"
  vlm:
    url: "http://127.0.0.1:8000"
    endpoint: "api/v1/workers/vlm-faiss-composed-retrieval"
  aesthetic:
    url: "http://127.0.0.1:8000"
    endpoint: "api/v1/workers/aesthetic"
```

Do not point a worker at `/api/v1/retrieval/<method>`. Those are proxy routes;
using one as its own upstream creates a recursive request loop.

Save those values in local `config/retrieval_methods.yaml`, then upload it
from a different WSL terminal:

```bash
colab upload -s "$SESSION_NAME" \
  config/retrieval_methods.yaml \
  /content/config/retrieval_methods.yaml
```

Use an Ngrok URL only when a worker is on another machine or you need external
access. Ensure `.env` contains `NGROK_TOKEN`, then run this from WSL while the
Uvicorn console remains open:

```bash
colab exec -s "$SESSION_NAME" --timeout 600 -f start_ngrok.py
```

It prints `Public URL: https://...`. Use that URL in place of
`http://127.0.0.1:8000` above, re-upload the YAML, and retain the server
console. Use the base URL only: do not append `/docs` or an endpoint path.

Run the benchmark through the local checkpoint driver (outside the Colab
console). It completes one scene remotely, creates a cumulative archive, and
downloads that archive before starting the next scene:

```bash
python scripts/run_benchmark_colab.py \
  --session "$SESSION_NAME" \
  --num-scenes 10 \
  --num-outfits 100 \
  --methods clip aesthetic vlm image_edit \
  --output-dir results/benchmark/colab-checkpoints-datazip-all
```

This evaluates all four configured methods. Each worker must support the
multipart `candidate_names` form field and return all 100 requested outfits
exactly once.

Checkpoints are saved locally under
`results/benchmark/colab-checkpoints-datazip-all/`:

- `scene-001.zip`, `scene-002.zip`, etc. are cumulative snapshots downloaded
  immediately after each scene finishes.
- `latest.zip` is the automatic resume source.
- `final.zip` includes `summary.json` and `per_scene.csv` after evaluation.

If the command or Colab VM stops, prepare a new session if necessary and rerun
the exact same command. The driver uploads `latest.zip`, restores it into the
new VM, and skips completed scene indices. Within a scene, rankings are written
after each worker response and Gemini judgments are append-only, so retrying is
also safe if interruption occurs before that scene's ZIP is downloaded. Keep
the same scene count, outfit count, seed, model, batch size, and method list;
the driver rejects an incompatible local checkpoint.

The driver does not release the runtime automatically. Always stop it when the
final archive has downloaded:

```bash
colab stop -s "$SESSION_NAME"
```

Do not use `run_colab.sh` for this foreground benchmark workflow: that script starts the FastAPI server and occupies the terminal. Use it separately when provisioning a retrieval worker, then run the benchmark from another terminal or session.
