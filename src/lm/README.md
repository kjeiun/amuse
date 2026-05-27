# Language Model Pretraining

This directory contains the language model pretraining code used for FineWeb-style Llama experiments.

## Quick Start

Run AMUSE on a 124M Llama-style model:

```bash
bash scripts/lm/124m/amuse.sh
```

Run this command from the repository root.

Set `YOUR_DATASET_DIR` in the script to the root directory used by the FineWeb loader.

## AMUSE Arguments

The language model AMUSE path is selected with `--opt amuse` in [`main.py`](main.py).

- `--lr`: base learning rate used for both AMUSE parameter groups.
- `--beta1`: initial interpolation factor.
- `--beta2`: second-moment coefficient for the Adam-style fallback group. AMUSE fixes this to `0.999`.
- `--momentum`: momentum used for the Muon-style hidden-matrix group. AMUSE fixes this to `0.95`.
- `--warmup_steps`: AMUSE warmup length when `--scheduler none`.
- `--rho`: controls how fast `beta1` moves toward `1` after warmup.
- `--weight_decay`: decoupled weight decay on the anchor parameters.
- `--weight_decay_at_y`: optional decay applied while parameters are still in `y` form.
- `--amuse_aux_opt`: auxiliary update type used for non-Muon parameters and Muon scaling. The default is `adamw`.

## Parameter Grouping

AMUSE is instantiated with two groups in [`main.py`](main.py).

Group 1: Muon-style hidden matrix group, `use_muon=True`

- all remaining parameters with `ndim >= 2` that were not assigned to the fallback group

In practice this is where the transformer hidden-layer matrices go, and this is the group that uses the Muon-style update inside AMUSE.

This group uses:

- `lr`
- `momentum=args.momentum` (`0.95` by default)
- `aux_update_type=args.amuse_aux_opt` (`adamw` by default)
- `weight_decay`

Group 2: Adam-style fallback, `use_muon=False`

- token embeddings such as `wte`
- position embeddings such as `wpe` when present
- scalar and vector parameters with `ndim < 2`
- `lm_head.weight`

This group uses:

- `lr`
- `beta2`
- `eps=1e-10`
- `update_type=args.amuse_aux_opt` (`adamw` by default)
- `weight_decay`


## Train and Eval Behavior

AMUSE follows the schedule-free style train/eval transition used in this codebase.

- before optimizer steps, the optimizer is expected to be in train mode
- during evaluation, the trainer switches AMUSE with `opt.eval()`

## Notes

- The AMUSE branch effectively expects `--scheduler none` so optimizer warmup is active.
- For multi-GPU runs, adjust `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` to match your machine.

## Codebase

The language modeling code builds on the setup from [Benchmarking Optimizers for LLM Pretraining](https://arxiv.org/abs/2509.01440) and [epfml/llm-optimizer-benchmark](https://github.com/epfml/llm-optimizer-benchmark).
