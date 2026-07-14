
<h1 align="center">AMUSE</h1>

<p align="center">
  <strong>AMUSE: Anytime Muon with Stable Gradient Evaluation</strong>
</p>

<p align="center">
  Jueun Kim* · Baekrok Shin* · Jihun Yun · Beomhan Baek · Minhak Song · Chulhee Yun
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.22432"><img src="https://img.shields.io/badge/arXiv-2605.22432-b31b1b.svg" alt="arXiv"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/BibTeX-Citation-orange.svg" alt="BibTeX"></a>
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python">
</p>

## Method Overview

AMUSE combines Muon with Schedule-Free updates by maintaining three sequences:
the fast base sequence $Z_t$, the averaged sequence $X_t$, and the gradient-evaluation
point $Y_t$. At each step, AMUSE evaluates the gradient at

$$
Y_t = (1-\beta_t) Z_t + \beta_t X_t,
$$

where the interpolation coefficient increases after warmup as

$$
\beta_t =
\begin{cases}
\beta_{\mathrm{init}}, & t \le T_0, \\
1 - \left(\frac{T_0}{t}\right)^\rho (1-\beta_{\mathrm{init}}), & t > T_0.
\end{cases}
$$

The parameter $\rho$ controls how quickly the gradient-evaluation point shifts from
the fast Muon trajectory $Z_t$ toward the stable averaged trajectory $X_t$.

For matrix-valued hidden parameters, AMUSE applies Muon at $Y_t$:

$$
M_t = \mu M_{t-1} + (1-\mu)\nabla L(Y_t), \qquad
O_t = \mathrm{NewtonSchulz}(M_t),
$$

$$
Z_{t+1} = Z_t - \eta O_t,
\qquad
X_{t+1} = \left(1-\frac{1}{t+1}\right) X_t + \frac{1}{t+1} Z_{t+1}.
$$

Thus, AMUSE preserves Muon's rapid progress in early training while gradually
stabilizing the trajectory through Schedule-Free averaging. This preserves Muon's rapid progress while reducing valley-wall oscillations, enabling schedule-free and anytime training.

**Full paper abstract**:
> Modern deep learning commonly relies on AdamW with prescribed learning rate schedules, but recent works challenge both components: Schedule-Free optimization removes explicit schedules via iterate averaging, and Muon improves the update geometry by orthogonalizing momentum for matrix parameters. Despite Muon's strong empirical performance, its underlying mechanism remains partially understood.
> We study Muon through the river-valley loss landscape, where useful training progress occurs along a flat, low-curvature bulk subspace, while high-curvature dominant directions form steep valley walls that induce oscillations. We empirically show that while Muon's orthogonalization accelerates river progress by increasing the bulk component, it also amplifies dominant-direction noise, causing oscillatory trajectories.
> Building on this, we propose **Anytime MUon with Stable gradient Evaluation (AMUSE)**, which integrates Muon's rapid bulk progress with the stabilizing effect of Schedule-Free averaging. AMUSE uses a time-varying interpolation coefficient that initially evaluates gradients near the fast Muon sequence for rapid adaptation, then gradually shifts toward the stable averaged sequence to suppress valley-wall oscillations. As a result, AMUSE requires no learning rate schedules and supports anytime training.
> Across vision tasks and large language model pretraining, AMUSE consistently improves the performance-iteration Pareto frontier over (Schedule-Free) AdamW and Muon.


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

For language model pretraining, run AMUSE on a 124M Llama-style model with:

```bash
bash scripts/lm/124m/amuse.sh
```

Set `YOUR_DATASET_DIR` in the script to the root directory used by the FineWeb-100B loader.

For image classification, run AMUSE on CIFAR-10 with:

```bash
bash scripts/image/cifar10/amuse.sh
```

Other image experiments are available through:

```bash
bash scripts/image/cifar100/amuse.sh
bash scripts/image/svhn/amuse.sh
bash scripts/image/imagenet/amuse.sh
```

For ImageNet, set `YOUR_DATASET_DIR` in the corresponding script. See [`src/lm/README.md`](src/lm/README.md) and [`src/image/README.md`](src/image/README.md) for task-specific optimizer and parameter grouping details.



## Results

### Language Model Pretraining

AMUSE achieves the performance-iteration Pareto frontier in Llama-style pretraining on FineWeb-100B.

<p align="center">
  <img src="assets/fineweb_llama_124M.png" width="720" alt="FineWeb Llama 124M pretraining results">
</p>


The same trend holds across model scales.

<p align="center">
  <img src="assets/fineweb_llama_720m_13b.png" width="720" alt="FineWeb Llama scaling results for 720M and 1.3B models">
</p>


### Image Classification

AMUSE also performs strongly across standard image classification benchmarks.

<p align="center">
  <img src="assets/image_benchmark.png" width="720" alt="image classification results">
</p>



## Citation

```bibtex
@article{kim2026amuse,
  title={{AMUSE}: Anytime Muon with Stable Gradient Evaluation},
  author={Kim, Jueun and Shin, Baekrok and Yun, Jihun and Baek, Beomhan and Song, Minhak and Yun, Chulhee},
  journal={arXiv preprint arXiv:2605.22432},
  year={2026}
}
```
