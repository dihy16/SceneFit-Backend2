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

# 3. Extract dataset from Google Drive. Prefer the compact benchmark archive.
archive_candidates = [
    "/content/drive/MyDrive/VRetrieval/benchmark_data.zip",
    "/content/drive/MyDrive/VRetrieval/data.zip",
]
drive_data_path = next((path for path in archive_candidates if os.path.exists(path)), None)
if drive_data_path:
    print("Unzipping dataset from Google Drive...")
    # Do not retain outfits/scenes from an earlier full archive: that would
    # defeat the compact benchmark archive and make index construction slow.
    run_bash("rm -rf /content/data/bg /content/data/2d")
    # The archive supplies a fresh manifest.  Remove stale judgments and
    # rankings so they cannot be mixed with this new candidate pool.
    run_bash("rm -rf /content/results/benchmark/latest")
    run_bash(f"unzip -o -q {drive_data_path} -d /content")
else:
    print("WARNING: No benchmark_data.zip or data.zip found. Skipping unzip.")

# 3. Build the visual search index
print("Building visual search index...")
run_bash("python -m scripts.build_pe_index")

print("Setup completed successfully.")
