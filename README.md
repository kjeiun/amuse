# AMUSE

This repository is based on the [epfml/llm-optimizer-benchmark](https://github.com/epfml/llm-optimizer-benchmark) codebase and adapts it for experiments with **AMUSE**.

## Setup

```bash
conda create -n amuse python=3.10
conda activate amuse
pip install -r requirements.txt
```

## Run AMUSE

The main example in this repository is [`scripts/124m/amuse.sh`](scripts/124m/amuse.sh).
It is the 124M AMUSE training run used on **FineWeb 100B**.

Before running it, set your dataset path:

```bash
export DATASET_DIR=/path/to/datasets
```

Then launch:

```bash
bash scripts/124m/amuse.sh
```

The current 124M example runs a 124M Llama-style model on **FineWeb 100B** with:

- `--opt amuse`
- `--nproc_per_node=8`
- `--dataset fineweb`
- `--n_embd 768 --n_head 12 --n_layer 12`
- `--batch_size 32 --acc_steps 8`

In the code, `FineWeb 100B` is launched through the `fineweb` dataset path together with `--datasets_dir $DATASET_DIR`.

You can also run the same FineWeb 100B setup directly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 ./src/main.py \
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
```

## AMUSE Arguments

The AMUSE path is selected with `--opt amuse`. The main AMUSE-specific or AMUSE-relevant arguments used by the code are:

- `--lr`: base learning rate used for both parameter groups.
- `--beta1`: initial interpolation factor between the training weights `y` and anchor weights `z`.
- `--beta2`: second-moment coefficient for the Adam-style fallback group.
- `--momentum`: momentum used for the Muon-style hidden-matrix group.
- `--warmup_steps`: required by AMUSE. In this codepath, AMUSE uses optimizer warmup when `--scheduler none`.
- `--rho`: controls how fast `beta1` moves toward `1` after warmup.
- `--weight_decay`: decoupled weight decay on the anchor parameters.
- `--weight_decay_at_y`: optional decay applied while parameters are still in `y` form.

Implementation references:

- optimizer: [`src/optim/AMUSE.py`](/home/jueun/llm-optimizer-benchmark/src/optim/AMUSE.py:50)
- optimizer construction: [`src/main.py`](/home/jueun/llm-optimizer-benchmark/src/main.py:280)
- config args: [`src/config/base.py`](/home/jueun/llm-optimizer-benchmark/src/config/base.py:85)

## Parameter Grouping

AMUSE is instantiated with two groups in [`src/main.py`](/home/jueun/llm-optimizer-benchmark/src/main.py:280).

Group 1: Adam-style fallback, `use_muon=False`

- token embeddings such as `wte`
- position embeddings such as `wpe` when present
- scalar and vector parameters with `ndim < 2`
- `lm_head.weight`

This group uses:

- `lr`
- `beta2`
- `eps=1e-10`
- `weight_decay`

Group 2: Muon-style hidden matrix group, `use_muon=True`

- all remaining parameters with `ndim >= 2` that were not assigned to the fallback group

In practice this is where the transformer hidden-layer matrices go, and this is the group that uses the Muon-style update inside AMUSE.

This group uses:

- `lr`
- `momentum`
- `weight_decay`

The grouping logic in code is:

```python
embed_params = [
    p for n, p in base_model.named_parameters()
    if ("embed" in n or "wte" in n or "wpe" in n)
    and not (hasattr(base_model, "lm_head") and p is base_model.lm_head.weight)
]
scalar_params = [p for p in base_model.parameters() if p.ndim < 2]
head_params = [base_model.lm_head.weight] if hasattr(base_model, "lm_head") else []
assigned_params = embed_params + scalar_params + head_params
hidden_matrix_params = [
    p for p in base_model.parameters()
    if p.ndim >= 2 and id(p) not in {id(x) for x in assigned_params}
]
```

## Train and Eval Behavior

AMUSE follows the schedule-free style train/eval transition used in this codebase.

- before optimizer steps, the optimizer is expected to be in train mode
- during evaluation, the code switches AMUSE with `opt.eval()`
- after evaluation, training resumes with `opt.train()`

This behavior is wired in [`src/optim/base.py`](/home/jueun/llm-optimizer-benchmark/src/optim/base.py:564).

## Notes

- `AMUSE` requires `warmup_steps > 0` in the optimizer implementation.
- In this repository, the AMUSE branch effectively expects `--scheduler none` so the optimizer warmup is active.
- For multi-GPU runs, adjust `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` to match your machine.
