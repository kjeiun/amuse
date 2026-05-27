set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "$REPO_ROOT"

# AMUSE 720M run on FineWeb 100B.
# Expects YOUR_DATASET_DIR to point to the dataset root used by the FineWeb loader.
torchrun --nproc_per_node=8 ./src/lm/main.py --config_format base --model llama --distributed_backend nccl \
  --n_embd 2048 --n_head 16 --n_layer 12 \
  --batch_size 62 --sequence_length 512 --acc_steps 32 \
  --dataset fineweb --datasets_dir YOUR_DATASET_DIR --iterations 16000 \
  --dropout 0.0 --warmup_steps 2000 --grad_clip 0.1 --seed 0 \
  --opt amuse --lr 0.01 --weight_decay 0.1 --scheduler none \
  --beta1 0.4 --rho 0.8 --beta2 0.999 --momentum 0.95 \
  --wandb --wandb_project YOUR_WANDB-PROJECT  --wandb_entity YOUR-WANDB-ENTITY \
  --eval_interval 200 --latest_ckpt_interval 1000
