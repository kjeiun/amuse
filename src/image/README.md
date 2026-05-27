# Image Classification

This directory contains the image classification experiments for CIFAR-10, CIFAR-100, SVHN, and ImageNet.

## Quick Start

Run AMUSE on CIFAR-10:

```bash
bash scripts/image/cifar10/amuse.sh
```

Run this command from the repository root.

Other image experiments are available in:

```bash
bash scripts/image/cifar100/amuse.sh
bash scripts/image/svhn/amuse.sh
bash scripts/image/imagenet/amuse.sh
```

For ImageNet, set `YOUR_DATASET_DIR` in the corresponding script.

## AMUSE Arguments

The image AMUSE path is selected with `--optimizer amuse` in [`train.py`](train.py).

- `--learning_rate`: learning rate for the Muon hidden-matrix group.
- `--sgd_learning_rate`: learning rate for the non-Muon fallback group. This can be tuned separately, but setting it equal to `--learning_rate` works well for AMUSE.
- `--beta`: initial interpolation factor.
- `--momentum`: momentum used for the Muon-style hidden-matrix group. AMUSE fix this to `0.95`.
- `--warmup_ratio`: fraction of total training steps used for AMUSE warmup.
- `--rho`: controls how fast `beta` moves toward `1` after warmup.
- `--decay`: weight decay used by both AMUSE groups.
- `--weight_decay_at_y`: optional decay applied while parameters are still in `y` form.
- `--amuse_aux_opt`: auxiliary update type used for non-Muon parameters and Muon scaling. The default is `sgd`.

## Parameter Grouping

AMUSE is instantiated with two groups in [`train.py`](train.py).

Group 1: Muon-style hidden matrix group, `use_muon=True`

- parameters with `ndim >= 2`, excluding `model.fc.weight`

This group uses:

- `lr=args.learning_rate`
- `momentum=args.momentum` (`0.95` by default)
- `aux_update_type=args.amuse_aux_opt` (`sgd` by default)
- `weight_decay`

Group 2: fallback group, `use_muon=False`

- scalar and vector parameters with `ndim < 2`
- classifier head weight `model.fc.weight`

The classifier head weight is not trained with Muon. When adding a new model, replace `model.fc.weight` with that model's classifier-head weight in the grouping logic.

This group uses:

- `lr=args.sgd_learning_rate`
- `update_type=args.amuse_aux_opt` (`sgd` by default)
- `weight_decay`

## Scheduler and Eval Behavior

- `sgd` and `muon` support `--scheduler cos`, `--scheduler warmupconst`, or `--scheduler none`.
- `sf-sgd` and `amuse` run schedule-free and use `scheduler=None`.
- `amuse` requires a positive warmup length in the optimizer implementation. If no warmup is used, especially for image experiments, set the window-growth start time to roughly 5% of the total training steps.
- During training, AMUSE calls `optimizer.train()` before optimizer steps to move parameters from `x` to `y`.
- During validation, AMUSE calls `optimizer.eval()` before measuring metrics to move parameters from `y` back to `x`.
- As in the original [Schedule-Free](https://github.com/facebookresearch/schedule_free) implementation, before AMUSE validation, `train.py` recomputes BatchNorm running statistics by running 50 train batches in `model.train()` mode with gradients disabled, then switches the model to `model.eval()` for validation.

```python
model.train()
optimizer.eval()
with torch.no_grad():
    for batch in itertools.islice(train_loader, 50):
        model(batch)
model.eval()
```
