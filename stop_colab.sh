#!/bin/bash

# Define the same session name used in run_colab.sh
SESSION_NAME="scenefit-backend"

echo "Stopping Colab session: $SESSION_NAME..."
colab stop -s "$SESSION_NAME"

echo "Colab session stopped successfully! GPU resources have been released."
