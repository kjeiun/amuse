
<h1 align="center">AMUSE</h1>

<p align="center">
  <strong>AMUSE: Anytime Muon with Stable Gradient Evaluation</strong>
</p>

<p align="center">
  Jueun Kim* · Baekrok Shin* · Jihun Yun · Beomhan Baek · Minhak Song · Chulhee Yun
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.22432">
    <img src="https://img.shields.io/badge/arXiv-2605.22432-b31b1b.svg" alt="arXiv">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
</p>

## Abstract

- AMUSE combines the fast progress of Muon with the stability of Schedule-Free optimization using a time-varying Schedule-Free momentum.

- From a river-valley perspective, Muon accelerates progress along flat bulk directions but can amplify oscillations along high-curvature dominant directions. AMUSE gradually moves gradient evaluation from the fast Muon trajectory toward a stable averaged trajectory, reducing oscillations while preserving rapid early progress.

- Across vision tasks and language model pretraining, AMUSE improves the performance-iteration Pareto frontier over AdamW, Schedule-Free AdamW, and Muon.

<details open>
<summary><strong>Full paper abstract</strong></summary>

Modern deep learning commonly relies on AdamW with prescribed learning rate schedules, but recent works challenge both components: Schedule-Free optimization removes explicit schedules via iterate averaging, and Muon improves the update geometry by orthogonalizing momentum for matrix parameters. Despite Muon's strong empirical performance, its underlying mechanism remains partially understood.
We study Muon through the river-valley loss landscape, where useful training progress occurs along a flat, low-curvature bulk subspace, while high-curvature dominant directions form steep valley walls that induce oscillations. We empirically show that while Muon's orthogonalization accelerates river progress by increasing the bulk component, it also amplifies dominant-direction noise, causing oscillatory trajectories.
Building on this, we propose **Anytime MUon with Stable gradient Evaluation (AMUSE)**, which integrates Muon's rapid bulk progress with the stabilizing effect of Schedule-Free averaging. AMUSE uses a time-varying interpolation coefficient that initially evaluates gradients near the fast Muon sequence for rapid adaptation, then gradually shifts toward the stable averaged sequence to suppress valley-wall oscillations. As a result, AMUSE requires no learning rate schedules and supports anytime training.
Across vision tasks and large language model pretraining, AMUSE consistently improves the performance-iteration Pareto frontier over (Schedule-Free) AdamW and Muon.

</details>


## Repository Structure

```text
amuse/
├── src/lm/       # language model pretraining experiments
├── src/image/    # vision/image experiments
├── src/optim/    # AMUSE and optimizer implementations
├── scripts/      # launch scripts
└── assets/       # figures and result plots
```


## Installation

```bash
conda create -n amuse python=3.10
conda activate amuse
pip install -r requirements.txt
```

## Quick Start

### Large Language Model Pretraining

Run AMUSE on a 124M Llama-style model:

```bash
bash scripts/lm/124m/amuse.sh
```

Set `YOUR_DATASET_DIR` in the script to the root directory used by the FineWeb-100B loader.

### Image Classification
Run AMUSE on CIFAR-10:
```bash
bash scripts/image/cifar10/amuse.sh
```

Other image experiments are available in:
```bash
bash scripts/image/cifar100/amuse.sh
bash scripts/image/svhn/amuse.sh
bash scripts/image/imagenet/amuse.sh
```

For ImageNet, set `YOUR_DATASET_DIR` in the corresponding script.



## Results

### Language Model Pretraining

AMUSE achieves the performance-iteration Pareto frontier in Llama-style pretraining on FineWeb-100B.

<p align="center">
  <img src="assets/fineweb_llama_124M.png" width="720" alt="FineWeb Llama 124M pretraining results">
</p>

<p align="center">
  <em>FineWeb Llama 124M pretraining.</em>
</p>

The same trend holds across model scales.

<p align="center">
  <img src="assets/fineweb_llama_720m_1b.png" width="720" alt="FineWeb Llama scaling results for 720M and 1B models">
</p>

<p align="center">
  <em>FineWeb Llama pretraining across 720M and 1B models.</em>
</p>



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
