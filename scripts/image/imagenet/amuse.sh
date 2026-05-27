#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "$REPO_ROOT"

# Expects YOUR_DATASET_DIR to point to the dataset root.
python ./src/image/train.py \
  --dataset imagenet --arch resnet50 --data_path YOUR_DATASET_DIR \
  --epochs 100 --batch-size 256 --warmup_ratio 0.05 --manualSeed 0 \
  --optimizer amuse --learning_rate 0.5 --sgd_learning_rate 0.5 \
  --decay 0.0005 --momentum 0.95 --beta 0.6 --rho 0.8 --scheduler none \
  --wandb --wandb_project YOUR_WANDB-PROJECT --wandb_entity YOUR-WANDB-ENTITY
