
set -euo pipefail

TARGET_DIR="./src/anemoi-house/replay/atmosphere-subsampled/p0/training"

mkdir -p "$TARGET_DIR"

cp ./src/submit_training_aml.sh "$TARGET_DIR/"

echo "Copied submit_training_aml.sh to $TARGET_DIR" 

