set -euo pipefail

# AMUSE 124M run on FineWeb 100B.
# Expects DATASET_DIR to point to the dataset root used by the FineWeb loader.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nproc_per_node=8 ./src/main.py  \
  --config_format base \
  --model llama \
  --distributed_backend nccl \
  --n_embd 768 \
  --n_head 12 \
  --n_layer 12 \
  --batch_size 32 \
  --sequence_length 512 \
  --acc_steps 8 \
  --dataset fineweb \
  --datasets_dir $DATASET_DIR \
  --iterations 16000 \
  --dropout 0.0 \
  --warmup_steps 6000 \
  --grad_clip 0.5 \
  --seed 0 \
  --opt amuse \
  --lr 0.01 \
  --weight_decay 0.05 \
  --beta1 0.6 \
  --rho 0.8 \
  --beta2 0.999 \
  --momentum 0.95 \
  --scheduler none \
  --eval_interval 115 \
  --latest_ckpt_interval 0 
