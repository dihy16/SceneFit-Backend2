# Gemma Retrieval Benchmark

This benchmark treats each scene as a retrieval query and uses one shared outfit candidate pool. `gemma-4-31b-it`, hosted by the Gemini API, first assigns a graded relevance score from 1 (unsuitable) to 5 (excellent) to every scene–outfit pair. With 10 scenes and 100 outfits, this creates 1,000 fixed labels. Retrieval workers then rank the same candidates; the evaluator reports nDCG@5/10 and mean relevance@5/10.

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
python scripts/benchmark.py pilot
python scripts/benchmark.py judge
python scripts/benchmark.py validate-judgments
python scripts/benchmark.py collect
python scripts/benchmark.py evaluate
```

`pilot` scores one pair without saving it as a benchmark label. `judge` sends ten outfits per request by default and writes the relevance matrix. Use `--batch-size 1` for independent pairwise requests. `validate-judgments` blocks retrieval unless all expected pairs match the manifest and evaluator configuration. `collect` calls worker URLs from `config/retrieval_methods.yaml` directly and requires each worker to return every manifest outfit exactly once.

Run the complete resumable workflow with:

```bash
python scripts/benchmark.py run --num-scenes 10 --num-outfits 100
```

Set every selected worker URL in `config/retrieval_methods.yaml` before using `run`. The combined command completes and validates every VLM label before contacting a retrieval worker. Each stage can also be resumed separately.

The output contains the data manifest and hashes, append-only `judgments.jsonl`, its evaluator metadata, cached rankings, `per_scene.csv`, and `summary.json`. Do not edit a manifest after judging begins; create a new run directory for a different sample.

## Running with a Google Colab GPU

Gemma 4 runs through Google's Gemini API and does not use the Colab GPU. For the
current light phase, a T4 runs only the CLIP and Aesthetic retrieval workers.
The VLM/SaMaG-R and ImageEdit methods can be run later on a separate GPU
machine using the saved benchmark artifacts.

If all retrieval workers already run on other machines, the benchmark session
itself does not benefit from a GPU; create a CPU session instead.

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

From the repository directory in WSL, create a T4 session and prepare it:

```bash
SESSION_NAME=scenefit-benchmark
colab new -s "$SESSION_NAME" --gpu T4
colab drivemount -s "$SESSION_NAME"
colab exec -s "$SESSION_NAME" --timeout 3600 -f setup_colab.py
colab upload -s "$SESSION_NAME" .env /content/.env
colab status -s "$SESSION_NAME"
```

If you reuse a session created before these changes were pushed, open its
console and run `cd /content && git pull --ff-only origin main` before starting
the benchmark.

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

The two reference-image checks are needed only for the later `image_edit`
phase. They may be skipped for this CLIP/Aesthetic-only Colab run.

## Judge First, Then Start Retrieval Workers

The benchmark evaluates retrieval *workers*, not just Gemma. Prepare the
manifest before starting the server so its vector database indexes only the
100 benchmark outfits instead of every image in `data/2d`. For this light
phase, selecting only CLIP and Aesthetic also skips the VLM-only PE-CLIP FAISS
index build:

```bash
python scripts/run_benchmark_colab.py \
  --session "$SESSION_NAME" \
  --num-scenes 10 \
  --num-outfits 100 \
  --methods clip aesthetic \
  --output-dir results/benchmark/checkpoints-light-gemma4 \
  --prepare-only
```

Use a fresh output directory for this evaluator. Do not reuse a checkpoint
created with Gemini 3.6 because labels from different judges must not be mixed.

Run the evaluator stage next from local WSL. Uvicorn is not needed yet:

```bash
python scripts/run_benchmark_colab.py \
  --session "$SESSION_NAME" \
  --num-scenes 10 \
  --num-outfits 100 \
  --methods clip aesthetic \
  --output-dir results/benchmark/checkpoints-light-gemma4 \
  --judge-only
```

The driver first makes one diagnostic pair call, which is printed but not
stored as a label. It then judges each scene and downloads a cumulative
`judge-scene-NNN.zip`. At batch size 10, the 1,000-pair matrix requires 100 API
calls. Repeat the same command after an interruption; completed labels resume.

After judging finishes, start the FastAPI server and keep it running for the
retrieval stage.
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
configuration uses its loopback address rather than Ngrok. Only the `clip` and
`aesthetic` entries are used in this phase; the other configured workers are
not contacted:

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

Run retrieval through the local checkpoint driver outside the Colab console.
It validates the complete relevance matrix before contacting any worker, then
downloads a cumulative archive after every retrieved scene:

```bash
python scripts/run_benchmark_colab.py \
  --session "$SESSION_NAME" \
  --num-scenes 10 \
  --num-outfits 100 \
  --methods clip aesthetic \
  --output-dir results/benchmark/checkpoints-light-gemma4 \
  --retrieve-only
```

This evaluates CLIP and Aesthetic only. Each worker must support the
multipart `candidate_names` form field and return all 100 requested outfits
exactly once.

The local driver prints a heartbeat every 30 seconds while Colab is executing
a remote command. The remote benchmark logs `[COLLECT]` messages for worker
requests and cache hits, `[JUDGE]` messages for each Gemma batch, and
`[GEMINI]` messages when an API attempt fails and will be retried. Colab CLI
may buffer remote output until the command finishes, so the local heartbeat is
the indication that the remote command is still active.

Checkpoints are saved locally under
`results/benchmark/checkpoints-light-gemma4/`:

- `pilot.zip` records that the diagnostic call succeeded; it contains no pilot label.
- `judge-scene-001.zip`, etc. are cumulative judgment snapshots.
- `scene-001.zip`, `scene-002.zip`, etc. are cumulative snapshots downloaded
  immediately after each retrieval scene finishes.
- `latest.zip` is the automatic resume source.
- `final.zip` includes `summary.json` and `per_scene.csv` after evaluation.

Keep `final.zip`. It is the handoff artifact for the later Vast.ai VLM and
ImageEdit phase: it preserves the exact manifest, Gemma judgments, and the
completed CLIP/Aesthetic rankings.

If the command or Colab VM stops, prepare a new session if necessary and rerun
the exact same command. The driver uploads `latest.zip`, restores it into the
new VM, and skips completed scene indices. Within a scene, rankings are written
after each worker response and Gemma judgments are append-only, so retrying is
also safe if interruption occurs before that scene's ZIP is downloaded. Keep
the same scene count, outfit count, seed, model, batch size, and method list;
the driver rejects an incompatible local checkpoint. Resume with
`--judge-only` or `--retrieve-only` for the interrupted stage.

The driver does not release the runtime automatically. Always stop it when the
final archive has downloaded:

```bash
colab stop -s "$SESSION_NAME"
```

Do not use `run_colab.sh` for this foreground benchmark workflow: that script starts the FastAPI server and occupies the terminal. Use it separately when provisioning a retrieval worker, then run the benchmark from another terminal or session.
