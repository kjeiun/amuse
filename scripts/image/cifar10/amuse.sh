#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "$REPO_ROOT"

python ./src/image/train.py \
  --dataset cifar10 --arch wideresnet --data_path ./data \
  --epochs 300 --batch-size 128 --warmup_ratio 0.05 --manualSeed 0 \
  --optimizer amuse --learning_rate 0.2 --sgd_learning_rate 0.2 \
  --decay 0.02 --momentum 0.95 --beta 0.8 --rho 0.3 --scheduler none \
  --wandb --wandb_project YOUR_WANDB-PROJECT --wandb_entity YOUR-WANDB-ENTITY
