#!/usr/bin/env bash

# This script copies the AzureML submission script to the default replay path.
# It ensures the target directory exists before copying.

set -euo pipefail

TARGET_DIR="anemoi-house/replay/atmosphere-subsampled/p0/training"

mkdir -p "$TARGET_DIR"

cp ./src/submit_training_aml.sh "$TARGET_DIR/"

echo "Copied submit_training_aml.sh to $TARGET_DIR" 

