import os
import subprocess

def run_bash(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

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

# 3. Extract dataset from Google Drive
drive_data_path = "/content/drive/MyDrive/VRetrieval/data.zip"
if os.path.exists(drive_data_path):
    print("Unzipping dataset from Google Drive...")
    run_bash(f"unzip -o -q {drive_data_path} -d /content/app")
else:
    print(f"WARNING: {drive_data_path} not found. Skipping unzip.")

# 3. Build the visual search index
print("Building visual search index...")
run_bash("python -m app.indexing.build_pe_index")

print("Setup completed successfully.")
