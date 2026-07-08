#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

SESSION_NAME=${1:-"scenefit-backend"}

# Check if google-colab-cli is installed
if ! command -v colab &> /dev/null; then
    exit 1
fi

# 1. Ensure Colab session is active (GPU T4 default)
if ! colab sessions | grep -q "$SESSION_NAME"; then
    colab new -s "$SESSION_NAME" --gpu T4
fi

# 2. Mount Google Drive (handles interactive auth)
colab drivemount -s "$SESSION_NAME"

# 3. Upload local .env if it exists
if [ -f .env ]; then
    colab upload -s "$SESSION_NAME" .env /content/.env
fi

# 5. Execute the setup script on Colab
colab exec -s "$SESSION_NAME" -f setup_colab.py

# 6. Execute the Ngrok tunnel setup script on Colab
colab exec -s "$SESSION_NAME" -f start_ngrok.py

# 7. Start the backend server using run.sh and stream logs
cat << 'EOF' | colab exec -s "$SESSION_NAME"
import subprocess
import sys

print("Starting backend server...")
process = subprocess.Popen(
    "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in process.stdout:
    print(line, end="")
    sys.stdout.flush()

process.wait()
EOF
