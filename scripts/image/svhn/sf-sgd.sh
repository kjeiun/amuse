#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "$REPO_ROOT"

python ./src/image/train.py \
  --dataset svhn --arch resnet3-96 --data_path ./data \
  --epochs 300 --batch-size 32 --warmup_ratio 0.05 --manualSeed 0 \
  --optimizer sf-sgd --learning_rate 1.0 --decay 0.0002 \
  --beta 0.9 --scheduler none \
  --wandb --wandb_project YOUR_WANDB-PROJECT --wandb_entity YOUR-WANDB-ENTITY
