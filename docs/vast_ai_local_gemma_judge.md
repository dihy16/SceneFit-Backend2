# Create Pairwise Ground Truth with 4-bit Gemma on Vast.ai

This pipeline runs the official Hugging Face `google/gemma-4-26B-A4B-it`
checkpoint directly in Transformers. BitsAndBytes dynamically loads its weights
in 4-bit NF4. It does **not** use llama.cpp, a local HTTP server, Gemini API, or
the four retrieval workers.

Vast.ai produces only the pairwise ground truth. Download that directory, then
evaluate the existing `clip`, `aesthetic`, `vlm`, and `image_edit` rankings on
the local computer.

Current Vast.ai connection:

```text
ssh -p 36457 root@69.176.92.121
```

You need a Hugging Face account that has accepted Gemma's model licence. Do not
put its access token in this repository or paste it into a command history.

## 1. Upload the only required inputs

Run in local PowerShell:

```powershell
cd "D:\10 Personal\10.11 Projects\SceneFit-Backend2"

ssh -p 36457 root@69.176.92.121 `
  "mkdir -p /workspace/artifacts/camviews"

scp -P 36457 `
  ".\storage\data.zip" `
  ".\results\benchmark\all-methods-evaluation\manifest.json" `
  root@69.176.92.121:/workspace/artifacts/

scp -P 36457 `
  ".\data\bg\CamView_01_Default.png" `
  ".\data\bg\CamView_Capture_02.png" `
  ".\data\bg\CamView_Capture_03.png" `
  ".\data\bg\CamView_Current.png" `
  root@69.176.92.121:/workspace/artifacts/camviews/
```

Do not upload `storage/final.zip` or `rankings/`: the judge needs neither.

## 2. Install the benchmark and restore its images

Connect to Vast.ai:

```powershell
ssh -p 36457 root@69.176.92.121
```

Run these commands on Vast.ai:

```bash
cd /workspace

if [ -d SceneFit-Backend2/.git ]; then
  cd SceneFit-Backend2
  git pull
else
  git clone https://github.com/dihy16/SceneFit-Backend2.git
  cd SceneFit-Backend2
fi

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade 'transformers>=5.0' accelerate bitsandbytes

unzip -o /workspace/artifacts/data.zip -d .
cp /workspace/artifacts/camviews/*.png data/bg/
mkdir -p results/benchmark/all-methods-evaluation
cp /workspace/artifacts/manifest.json \
  results/benchmark/all-methods-evaluation/manifest.json
```

Log in to Hugging Face once. Paste the token only into the password prompt:

```bash
huggingface-cli login
```

Verify all judge images are present:

```bash
python -c 'import json, pathlib; m=json.loads(pathlib.Path("results/benchmark/all-methods-evaluation/manifest.json").read_text()); missing=[x["path"] for x in m["scenes"]+m["outfits"] if not pathlib.Path(x["path"]).is_file()]; assert not missing, f"Missing manifest files: {missing}"; print("All 10 scenes and 100 outfits are ready")'
```

## 3. Pilot the model

The first run downloads the roughly 16 GB model and then loads it on the GPU.
Run this in the same Vast.ai terminal:

```bash
cd /workspace/SceneFit-Backend2
source .venv/bin/activate

python scripts/benchmark.py pilot \
  --judge-backend transformers-bnb-4bit \
  --model google/gemma-4-26B-A4B-it \
  --manifest results/benchmark/all-methods-evaluation/manifest.json \
  --max-attempts 3
```

Continue only after the command prints one `comparison` containing all five
criteria. A 48 GB GPU is recommended. This backend has no server process to
start or keep running.

## 4. Generate or resume the ground truth

```bash
python scripts/benchmark.py judge \
  --judge-backend transformers-bnb-4bit \
  --model google/gemma-4-26B-A4B-it \
  --manifest results/benchmark/all-methods-evaluation/manifest.json \
  --judge-dir results/benchmark/pairwise_ground_truth_gemma4_26b_bnb_nf4 \
  --pairs-per-request 1 \
  --concurrency 1 \
  --max-attempts 3
```

`--concurrency 1` is intentional. One in-process model is shared by the
requests; multiple concurrent generations would compete for the same GPU memory
and do not make this backend faster. If interrupted, rerun the exact command:
saved original and mirrored responses are skipped automatically.

## 5. Validate and download the ground truth

On Vast.ai, after judging completes:

```bash
python scripts/benchmark.py validate-judge \
  --judge-backend transformers-bnb-4bit \
  --manifest results/benchmark/all-methods-evaluation/manifest.json \
  --judge-dir results/benchmark/pairwise_ground_truth_gemma4_26b_bnb_nf4 \
  --model google/gemma-4-26B-A4B-it
```

It should finish with:

```text
Pairwise judge complete: 3500 comparisons
```

Back on local PowerShell, download only the resulting judge artifacts:

```powershell
cd "D:\10 Personal\10.11 Projects\SceneFit-Backend2\results\benchmark"

scp -P 36457 -r `
  root@69.176.92.121:/workspace/SceneFit-Backend2/results/benchmark/pairwise_ground_truth_gemma4_26b_bnb_nf4 `
  .
```

After confirming the download, stop the Vast.ai instance to stop billing.

## 6. Calculate all four method scores locally

No GPU, model download, or local judge server is needed for these commands:

```powershell
cd "D:\10 Personal\10.11 Projects\SceneFit-Backend2"

python scripts/benchmark.py validate-judge `
  --judge-backend transformers-bnb-4bit `
  --manifest results/benchmark/all-methods-evaluation/manifest.json `
  --judge-dir results/benchmark/pairwise_ground_truth_gemma4_26b_bnb_nf4 `
  --model google/gemma-4-26B-A4B-it

python scripts/benchmark.py evaluate `
  --manifest results/benchmark/all-methods-evaluation/manifest.json `
  --rankings-dir results/benchmark/all-methods-evaluation/rankings `
  --judge-dir results/benchmark/pairwise_ground_truth_gemma4_26b_bnb_nf4 `
  --output-dir results/benchmark/pairwise_evaluation_gemma4_26b_bnb_nf4 `
  --protocol pairwise-elo-v1
```

The second command reports metrics for all four existing method ranking files.
