# Finish the benchmark on Vast.ai with ImageEdit

This runbook adds `image_edit` rankings to the completed `clip` and
`aesthetic` benchmark. It reuses the exact manifest and existing judgments; do
not create a new manifest or judge the data again.

## Bring these artifacts

This repository already has the two needed archives:

- `data/final.zip` is the completed 10-scene, 100-outfit checkpoint. It
  contains the manifest, local-VLM judgments, and CLIP/Aesthetic rankings.
- `data/data.zip` is the source dataset archive. It contains the outfit images,
  six screenshot backgrounds, clothing metadata, and both ImageEdit reference
  images. It does **not** contain the four CamView backgrounds selected by this
  manifest; upload those separately after restoring the archive.

Use these files together. If a replacement compact archive is needed, generate
it from the same repository revision, scene/outfit counts, and seed:

```bash
python scripts/package_benchmark_data.py --output benchmark_data.zip \
  --num-scenes 10 --num-outfits 100 --seed 42
```

Never commit these archives or `.env` files. Do not edit `manifest.json` after
judging starts.

## Create the Vast.ai worker

Use a CUDA/PyTorch image with SSH access and persistent disk. Flux.2-klein-9B
is a large BF16 pipeline: start with a 48 GB GPU; a 24 GB GPU can run out of
memory depending on the installed models and allocator. Before renting, accept
the Flux Hugging Face license and obtain a valid `HF_TOKEN`.

ImageEdit asks a VLM for an outfit description, releases the VLM, loads Flux,
edits/crops the reference image, then searches the local candidate index. Set
`VLM_BASE_URL` to a worker that exposes `/api/v1/workers/vlm-suggest-outfit`.
For a single-worker setup, use `http://127.0.0.1:8000`; otherwise use the base
URL of a reachable dedicated VLM worker.

Upload the artifacts from your computer, replacing the SSH address with the
one displayed by Vast.ai:

```bash
scp data/data.zip data/final.zip \
  root@<VAST_HOST>:/workspace/artifacts/
ssh root@<VAST_HOST>
```

On the instance, clone the exact commit used to create `final.zip` and restore
the run:

```bash
cd /workspace
git clone <YOUR_REPOSITORY_URL> SceneFit-Backend2
cd SceneFit-Backend2
git checkout <THE_SAME_COMMIT_AS_COLAB>

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

unzip -o /workspace/artifacts/data.zip -d .
mkdir -p results/benchmark/latest
unzip -o /workspace/artifacts/final.zip -d results/benchmark/latest
```

The saved manifest also requires these four backgrounds, which are intentionally
outside `data/data.zip`. From your local PowerShell after the restore above,
upload them directly into the cloned repository:

```powershell
scp -P <PORT> `
  "data/bg/CamView_01_Default.png" `
  "data/bg/CamView_Capture_02.png" `
  "data/bg/CamView_Capture_03.png" `
  "data/bg/CamView_Current.png" `
  root@<IP>:/workspace/SceneFit-Backend2/data/bg/
```

On the Vast.ai instance, verify all ten manifest backgrounds exist:

```bash
test "$(find data/bg -maxdepth 1 -type f | wc -l)" -eq 10
```

Create `.env` without quotes around tokens. `IMAGEROUTER_API_KEY` is required
by the current application settings even though the Flux benchmark endpoint
does not call ImageRouter; a non-secret placeholder is sufficient for this
Flux-only run:

```dotenv
IMAGEROUTER_API_KEY=not_used_by_flux_benchmark
HF_TOKEN=hf_your_token
```

Do not put `VLM_BASE_URL` in `.env`: the current settings schema rejects that
undeclared setting. Export it in the shell before starting Uvicorn instead.
For an external VLM, export its base URL in place of `http://127.0.0.1:8000`.
The Flux worker uses its local vector database in this path, so
`VECTOR_DB_BASE_URL` is not required.

Validate inputs and model access before the long run:

```bash
nvidia-smi
test -f data/ref_images/man.png
test -f data/ref_images/woman.png
test -f results/benchmark/latest/manifest.json
test -f results/benchmark/latest/judgments.jsonl
python scripts/benchmark.py smoke
python scripts/prefetch_flux.py
```

Fix Hugging Face authorization before continuing. If prefetching causes CUDA
out-of-memory, stop the instance and rent a larger GPU.

## Start the benchmark-mode worker

In terminal 1, run the API with the restored manifest. This constrains the
local vector index to the benchmark's 100 outfits.

```bash
cd /workspace/SceneFit-Backend2
source .venv/bin/activate
export VLM_BASE_URL=http://127.0.0.1:8000
BENCHMARK_MANIFEST="$PWD/results/benchmark/latest/manifest.json" \
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for Uvicorn to be running; first start should encode 100 outfits. Keep
this terminal open.

In terminal 2, make a dedicated benchmark config and ensure its ImageEdit
worker points at the concrete worker endpoint:

```bash
cd /workspace/SceneFit-Backend2
source .venv/bin/activate
cp config/retrieval_methods.yaml config/retrieval_methods.vastai-image-edit.yaml
nano config/retrieval_methods.vastai-image-edit.yaml
```

```yaml
  image_edit:
    url: "http://127.0.0.1:8000"
    endpoint: "api/v1/workers/image-edit-flux"
    description: "Image edit model retrieval"
```

Do not use `/api/v1/retrieval/image_edit`; it is a proxy route, and pointing it
to the same API causes recursive requests.

## Add ImageEdit rankings and evaluate

`final.zip` records the earlier checkpoint configuration with methods `clip`
and `aesthetic`. Therefore, do **not** run
`run_benchmark_runtime.py retrieve --methods image_edit`: its configuration
guard will correctly reject the changed method list. Use the lower-level
collector below. It retains existing ranking files and adds
`rankings/image_edit.json`.

```bash
cd /workspace/SceneFit-Backend2
source .venv/bin/activate

python scripts/benchmark.py validate-judgments \
  --manifest results/benchmark/latest/manifest.json \
  --judgments results/benchmark/latest/judgments.jsonl \
  --model local-agent-vlm-v1 \
  --batch-size 1

python scripts/benchmark.py collect \
  --manifest results/benchmark/latest/manifest.json \
  --judgments results/benchmark/latest/judgments.jsonl \
  --output-dir results/benchmark/latest/rankings \
  --config config/retrieval_methods.vastai-image-edit.yaml \
  --methods image_edit \
  --model local-agent-vlm-v1 \
  --batch-size 1

python scripts/benchmark.py evaluate \
  --manifest results/benchmark/latest/manifest.json \
  --judgments results/benchmark/latest/judgments.jsonl \
  --rankings-dir results/benchmark/latest/rankings \
  --output-dir results/benchmark/latest
```

Each request must rank every supplied candidate exactly once. The collector
saves after every scene, so rerunning the same `collect` command safely resumes
after a disconnection or failed request.

## Verify, download, and stop billing

```bash
python -m json.tool results/benchmark/latest/summary.json
test -f results/benchmark/latest/rankings/image_edit.json

cd /workspace/SceneFit-Backend2/results/benchmark/latest
zip -r /workspace/artifacts/benchmark-with-image-edit.zip .

# Run locally, not on the Vast.ai worker:
scp root@<VAST_HOST>:/workspace/artifacts/benchmark-with-image-edit.zip \
  results/benchmark/
```

The downloaded archive should include `summary.json`, `per_scene.csv`,
`rankings/image_edit.json`, the original manifest/judgments, and the prior
CLIP/Aesthetic rankings. Verify the local download, then stop or destroy the
Vast.ai instance to stop GPU charges.

## Troubleshooting

- **Reference image missing:** restore `data.zip`; both
  `data/ref_images/man.png` and `data/ref_images/woman.png` are required.
- **Hugging Face 401/403:** accept the Flux license for the account owning
  `HF_TOKEN`, then restart the worker.
- **VLM connection error:** set `VLM_BASE_URL` to a reachable base URL only;
  the service appends `/api/v1/workers/vlm-suggest-outfit` itself.
- **Out of memory:** choose a larger GPU. Flux and VLM are released between
  stages, but either model can still exceed a small card.
- **Exact manifest pool error:** confirm Uvicorn was started with
  `BENCHMARK_MANIFEST` and no proxy removes the `candidate_names` form field.
