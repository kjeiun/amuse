#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "$REPO_ROOT"

python ./src/image/train.py \
  --dataset cifar100 --arch densenet --data_path ./data \
  --epochs 300 --batch-size 64 --warmup_ratio 0.05 --manualSeed 0 \
  --optimizer amuse --learning_rate 0.5 --sgd_learning_rate 0.5 \
  --decay 0.002 --momentum 0.95 --beta 0.7 --rho 0.4 --scheduler none \
  --wandb --wandb_project YOUR_WANDB-PROJECT --wandb_entity YOUR-WANDB-ENTITY
