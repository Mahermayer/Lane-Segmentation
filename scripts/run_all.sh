#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CONFIG="${1:-configs/train/focal_dice.yaml}"

VARIANTS=(
  vanilla
  depthwise
  se_decoder
  aspp
  depthwise_se
  depthwise_aspp
  se_aspp
  full
)

for variant in "${VARIANTS[@]}"; do
  echo "Starting variant: ${variant}"
  OUTPUT_DIR="${ROOT_DIR}/outputs/${variant}"
  mkdir -p "${OUTPUT_DIR}"
  python3 "${ROOT_DIR}/scripts/train.py" \
    --model-config "${ROOT_DIR}/configs/model/${variant}.yaml" \
    --train-config "${ROOT_DIR}/${TRAIN_CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"
done

