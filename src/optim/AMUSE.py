import torch
import torch.optim

UPDATE_TYPES = {"muon", "adamw", "sgd"}
AUX_UPDATE_TYPES = {"adamw", "sgd"}


@torch.no_grad()
def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.bfloat16()
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


@torch.no_grad()
def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    aux_update_type: str = "adamw",
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    if update.ndim == 4:
        update = update.view(len(update), -1)

    update = zeropower_via_newtonschulz5(update)

    if aux_update_type == "adamw":
        # Scaling used in the AdamW-aux AMUSE setting.
        update *= 0.2 * max(update.size(0), update.size(1)) ** 0.5
    elif aux_update_type == "sgd":
        # Muon default scaling used when auxiliary layers are trained by SGD.
        update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    else:
        raise ValueError(
            f"Invalid AMUSE aux_update_type: {aux_update_type}. "
            "Expected one of {'adamw', 'sgd'}."
        )

    return update


class AMUSE(torch.optim.Optimizer):
    """
    AMUSE optimizer.

    State convention:
    - p stores y while training.
    - eval() converts y -> x using the current beta1.
    - train() converts x -> y using the current beta1.
    - state["z"] stores the anchor z.

    Hyperparameters:
    - beta1: initial y/x interpolation. During warmup beta1 is constant.
    - rho: controls how quickly beta1 approaches 1 after warmup.
      Higher rho pushes beta1 toward 1 faster, so y moves closer to x
      faster. Lower rho keeps y farther from x for longer.
    - r: polynomial power for the z/x averaging weights.
    - weight_decay: decoupled decay applied to z for Muon and SGD paths.
    - weight_decay_at_y: optional decay applied while p is still y.

    Parameter groups:
    - Matrix hidden-layer weights should use {"use_muon": True}. These use
      Muon momentum and Newton-Schulz orthogonalization.
    - Embeddings, scalar parameters, and output head weights should use
      {"use_muon": False}. By default these use the AdamW-style fallback.
    - Non-Muon groups can set {"update_type": "sgd"} to use the same AMUSE
      y/x/z schedule with a plain SGD z update instead of AdamW.
    - Muon groups can set {"aux_update_type": "sgd"} to use Muon's default
      scaling when the auxiliary layers are trained by SGD.
    - The AdamW-style fallback uses the group's beta2 value.
    - Each group should provide lr and weight_decay; Muon groups may also
      provide momentum.
    """
    def __init__(
        self,
        param_groups,
        *,
        weight_decay_at_y: float = 0.0,
        beta1: float = 0.9,
        weight_lr_power: float = 2.0,
        warmup_steps: int = 0,
        rho: float = 1.0,
        r: float = 0.0,
    ):
        if warmup_steps <= 0:
            raise ValueError("AMUSE requires warmup_steps > 0.")

        self.weight_decay_at_y = weight_decay_at_y
        self.beta1_init = float(beta1)
        self.weight_lr_power = weight_lr_power
        self.warmup_steps = int(warmup_steps)
        self.rho = float(rho)
        self.r = r
        self.train_mode = False

        super().__init__(param_groups, defaults={})
        for group in self.param_groups:
            group.setdefault("warmup_steps", self.warmup_steps)
            group.setdefault("k", 0)
            group.setdefault("weight_sum", 0.0)
            group.setdefault("use_muon", False)
            group.setdefault("weight_decay", 0.0)
            group.setdefault("beta1", self.beta1_init)

            update_type = (
                "muon" if group["use_muon"] else group.get("update_type", "adamw")
            )
            if update_type not in UPDATE_TYPES:
                raise ValueError(
                    f"Invalid AMUSE update_type: {update_type}. "
                    "Expected one of {'muon', 'adamw', 'sgd'}."
                )
            if update_type == "muon" and not group["use_muon"]:
                raise ValueError('AMUSE update_type="muon" requires use_muon=True.')

            group["update_type"] = update_type

            if update_type == "muon":
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("aux_update_type", "adamw")
                if group["aux_update_type"] not in AUX_UPDATE_TYPES:
                    raise ValueError(
                        f"Invalid AMUSE aux_update_type: {group['aux_update_type']}. "
                        "Expected one of {'adamw', 'sgd'}."
                    )
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                for p in group["params"]:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
            elif update_type == "adamw":
                group.setdefault("lr", 3e-4)
                group.setdefault("beta2", 0.999)
                group.setdefault("eps", 1e-10)
                for p in group["params"]:
                    state = self.state[p]
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(p)
            elif update_type == "sgd":
                group.setdefault("lr", 1.0)

            group["base_lr"] = group["lr"]

    def _compute_beta1(self, group, t, ckp1, warmup_steps):
        if t <= warmup_steps:
            if t == warmup_steps:
                group["c_warmup"] = ckp1
            return self.beta1_init

        c_warmup = group.get("c_warmup", 1.0 / warmup_steps)
        S_t = (ckp1 * (1.0 - c_warmup)) / (c_warmup * (1.0 - ckp1))
        return 1.0 - (S_t ** self.rho) * (1.0 - self.beta1_init)

    def _get_z(self, p):
        state = self.state[p]
        z = state.get("z")
        if z is None:
            z = state["z"] = torch.clone(p, memory_format=torch.preserve_format)
        return z

    def _apply_weight_decay_at_y(self, p, z, lr, beta1):
        if self.weight_decay_at_y == 0.0:
            return
        z.sub_(p, alpha=lr * self.weight_decay_at_y)
        p.sub_(p, alpha=lr * self.weight_decay_at_y * (1.0 - beta1))

    @torch.no_grad()
    def eval(self):
        if self.train_mode:
            for group in self.param_groups:
                beta1 = group.get("beta1", self.beta1_init)
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1.0 - 1.0 / beta1)
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if not self.train_mode:
            for group in self.param_groups:
                beta1 = group.get("beta1", self.beta1_init)
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1.0 - beta1)
        self.train_mode = True

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            raise Exception(
                "Optimizer was not in train mode when step is called. "
                "Please insert .train() and .eval() calls on the optimizer."
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            base_lr = group["base_lr"]
            k = group["k"]
            warmup_steps = group.get("warmup_steps", self.warmup_steps)
            if warmup_steps <= 0:
                raise ValueError("AMUSE requires warmup_steps > 0.")

            t = k + 1
            sched = min(1.0, t / warmup_steps)
            lr = base_lr * sched
            group["lr"] = lr

            # ckp1 is the new z-to-x averaging weight c_t.
            weight = (t ** self.r) * (lr ** self.weight_lr_power)
            future_weight_sum = group.get("weight_sum", 0.0) + weight
            ckp1 = weight / future_weight_sum if future_weight_sum > 0 else 1.0
            group["ckp1"] = ckp1
            group["weight_sum"] = future_weight_sum

            beta1 = self._compute_beta1(group, t, ckp1, warmup_steps)
            group["beta1"] = beta1
            group["r_t"] = ckp1 / ((1.0 - beta1) + beta1 * ckp1 + 1e-12)
            wd = group.get("weight_decay", 0.0)
            self.beta1 = beta1

            update_type = group["update_type"]

            if update_type == "muon":
                beta_m = group["momentum"]
                aux_update_type = group.get("aux_update_type", "adamw")
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    z = self._get_z(p)
                    self._apply_weight_decay_at_y(p, z, lr, beta1)

                    # y_t -> x_t, then update z, then rebuild y_{t+1}.
                    p.lerp_(end=z, weight=1.0 - 1.0 / beta1)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=beta_m,
                        aux_update_type=aux_update_type,
                        nesterov=True,
                    )
                    if wd != 0.0:
                        z.mul_(1.0 - lr * wd)
                    z.add_(update.reshape(p.shape), alpha=-lr)
                    p.lerp_(end=z, weight=ckp1)
                    p.lerp_(end=z, weight=1.0 - beta1)

            elif update_type == "adamw":
                beta2 = group.get("beta2", 0.999)
                eps = group.get("eps", 1e-10)
                bias_correction2 = 1.0 - beta2 ** t
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    
                    state = self.state[p]
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    z = self._get_z(p)
                    self._apply_weight_decay_at_y(p, z, lr, beta1)

                    # y_t -> x_t, then update z, then rebuild y_{t+1}.
                    p.lerp_(end=z, weight=1.0 - 1.0 / beta1)
                    v = state["exp_avg_sq"]
                    grad = p.grad
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = v.div(bias_correction2).sqrt_().add_(eps)
                    update = grad / denom
                    if wd != 0.0:
                        update = update.add(z, alpha=wd)
                    z.add_(update, alpha=-lr)
                    p.lerp_(end=z, weight=ckp1)
                    p.lerp_(end=z, weight=1.0 - beta1)

            elif update_type == "sgd":
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    z = self._get_z(p)
                    self._apply_weight_decay_at_y(p, z, lr, beta1)

                    # y_t -> x_t, then update z, then rebuild y_{t+1}.
                    p.lerp_(end=z, weight=1.0 - 1.0 / beta1)
                    if wd != 0.0:
                        z.mul_(1.0 - lr * wd)
                    z.add_(p.grad, alpha=-lr)
                    p.lerp_(end=z, weight=ckp1)
                    p.lerp_(end=z, weight=1.0 - beta1)

            else:
                raise ValueError(f"Invalid AMUSE update_type: {update_type}")

            group["k"] = k + 1

        return loss
