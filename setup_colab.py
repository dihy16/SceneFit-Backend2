import os
import subprocess

def run_bash(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"COMMAND FAILED: {command}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)

print("Starting Colab environment setup...")

# 1. Clean up, clone repository, and set permissions
run_bash("rm -rf /content/SceneFit-Backend2")
run_bash("git clone --branch main --recurse-submodules https://github.com/dihy16/SceneFit-Backend2.git")
run_bash("rsync -a /content/SceneFit-Backend2/ /content/")
run_bash("rm -rf /content/SceneFit-Backend2")
run_bash("chmod +x /content/run.sh")

# 2. Install dependencies
print("Installing dependencies...")
run_bash("pip install pyngrok ftfy 'rembg[gpu]' faiss-cpu 'qwen-vl-utils>=0.0.14' python-dotenv")
if os.path.exists("/content/requirements.txt"):
    run_bash("pip install -r /content/requirements.txt")

# 3. Extract the full dataset from Google Drive.
drive_data_path = "/content/drive/MyDrive/VRetrieval/data.zip"
if os.path.exists(drive_data_path):
    print("Unzipping dataset from Google Drive...")
    # Avoid mixing files or benchmark artifacts from an earlier run with the
    # archive that is about to be extracted.
    run_bash("rm -rf /content/data/bg /content/data/2d")
    run_bash("rm -rf /content/results/benchmark/latest")
    run_bash(f"unzip -o -q {drive_data_path} -d /content")
else:
    raise FileNotFoundError(
        f"Required dataset archive not found: {drive_data_path}. "
        "Mount Google Drive and upload VRetrieval/data.zip before setup."
    )

# Benchmark-mode Uvicorn builds an index containing only the outfits selected
# by BENCHMARK_MANIFEST. Building the general full-dataset index here would do
# duplicate work for the benchmark.
print("Deferring visual index construction to application startup.")

print("Setup completed successfully.")
