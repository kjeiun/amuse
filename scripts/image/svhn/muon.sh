#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "$REPO_ROOT"

python ./src/image/train.py \
  --dataset svhn --arch resnet3-96 --data_path ./data \
  --epochs 300 --batch-size 32 --warmup_ratio 0.05 --manualSeed 0 \
  --optimizer muon --learning_rate 0.05 --sgd_learning_rate 0.001 \
  --decay 0.002 --momentum 0.9 --scheduler cos \
  --wandb --wandb_project YOUR_WANDB-PROJECT --wandb_entity YOUR-WANDB-ENTITY
