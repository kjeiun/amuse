import argparse
import copy
import hashlib
import inspect
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

import config
import distributed
from data.utils import DataReader, get_dataset
from models.utils import get_model
from optim.adafactor import Adafactor
from optim.ademamix import AdEMAMix
from optim.adopt import ADOPT
from optim.AMUSE import AMUSE
from optim.base import train
from optim.lamb import Lamb
from optim.lion import Lion
from optim.mars import MARS
from optim.muon import CombinedScheduler, DistributedMuon, Muon
from optim.prodigy import Prodigy
from optim.schedule import cos_inf_schedule, wsd_schedule
from optim.schedulefree import AdamWScheduleFree, SGDScheduleFree
from optim.scion import Scion, ScionLight, scion_partitions
from optim.sign import Signum
from optim.soap import SOAP
from optim.sophia import SophiaG


class DMuonWarmupScheduler:
    """Cosine warmup-only scheduler matching the warmup phase of OneCycleLR."""

    def __init__(self, optimizer, warmup_steps, div_factor=1e2):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.div_factor = float(div_factor)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = 0

        if self.warmup_steps > 0:
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr / self.div_factor

    def _compute_lr(self, base_lr, step_idx):
        if self.warmup_steps <= 0:
            return base_lr
        if step_idx >= self.warmup_steps:
            return base_lr

        start_lr = base_lr / self.div_factor
        pct = step_idx / self.warmup_steps
        return base_lr + (start_lr - base_lr) * (1.0 + np.cos(np.pi * pct)) / 2.0

    def step(self):
        self.last_epoch += 1
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self._compute_lr(base_lr, self.last_epoch)

    def state_dict(self):
        return {
            "warmup_steps": self.warmup_steps,
            "div_factor": self.div_factor,
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict):
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.div_factor = float(state_dict["div_factor"])
        self.base_lrs = list(state_dict["base_lrs"])
        self.last_epoch = int(state_dict["last_epoch"])
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self._compute_lr(base_lr, self.last_epoch)


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )

    args, rem_args = parser.parse_known_args()

    final_args = config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )

    return final_args, parser


def main(args, parser):
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if args.full_eval_at is None:
        args.full_eval_at = []
    if getattr(args, "ema", False):
        args.exponential_weight_average = True

    # NOTE args.seed is offset per worker in get_adjusted_args_for_process
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))
    # torch.use_deterministic_algorithms(True)  # CUBLAS_WORKSPACE_CONFIG=:4096:8

    wandb_name = get_exp_name(args, parser, distributed_backend)
    exp_name, exp_dir = resolve_experiment_dir(args, wandb_name)
    if distributed_backend.is_master_process() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=vars(args),
            entity=args.wandb_entity,
        )
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("val/*", step_metric="iter")
        wandb.define_metric("lr", step_metric="iter")

    print(f"Starting Experiment: {wandb_name}")
    if exp_name != wandb_name:
        print(f"Filesystem Experiment Name: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")

    print(f"Loading dataset: '{args.dataset}'")
    datareaders = get_data_readers(args)

    model = get_model(args).to(
        args.device
    )  # todo: take care of initializing the model if args.use_pretrained != 'none'
    print(f"\nModel:\n{model}")

    model = distributed_backend.transform_model(model)

    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs(
        config=args
    )
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = (
                distributed_backend.translate_model_parameter_name_for_node(p_name)
            )
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    nonemb_param_cnt = (
        params_cnt
        - distributed_backend.get_raw_model(model).lm_head.weight.numel()
        - distributed_backend.get_raw_model(model).transformer.wte.weight.numel()
    )
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    print("number of non-embedding parameters: %.2fM" % (nonemb_param_cnt / 1e6,))
    if args.wandb and distributed_backend.is_master_process():
        wandb.log(
            {
                "parameters": params_cnt,
                "optimized_parameters": optimized_params_cnt,
                "non_embedding_parameters": nonemb_param_cnt,
            }
        )

    args.world_size = distributed_backend.get_world_size()

    if args.opt == "adamw":
        device_type = "cuda" if "cuda" in args.device else "cpu"
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            **extra_args,
        )
    elif args.opt == "soap":
        opt = SOAP(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            weight_decay=args.weight_decay,
            precondition_frequency=args.precondition_frequency,
            max_precond_dim=args.max_precond_dim,
            merge_dims=args.merge_dims,
            precondition_1d=args.precondition_1d,
            normalize_grads=args.normalize_grads,
            data_format=args.soap_data_format,
            correct_bias=args.correct_bias,
        )
    elif args.opt == "muon":
        param_list = (
            list(model.parameters())
            if args.distributed_backend is None
            else list(model.module.parameters())
        )
        opt = Muon(
            muon_params=param_list,
            lr=args.muon_lr_factor,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            adamw_params=None,
            adamw_lr=args.lr,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            adamw_wd=args.weight_decay,
        )
    elif args.opt == "d-muon":
        opt = DistributedMuon(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=1e-8,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "ademamix":
        opt = AdEMAMix(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.adema_beta3),
            alpha=args.adema_alpha,
            beta3_warmup=args.adema_beta3_warmup,
            alpha_warmup=args.adema_alpha_warmup,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "lion":
        opt = Lion(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "sf-adamw":
        sf_warmup_steps = 0 if args.scheduler != "none" else args.warmup_steps
        opt = AdamWScheduleFree(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            warmup_steps=sf_warmup_steps,
            r=args.schedulefree_r,
            weight_lr_power=args.weight_lr_power,
        )  # without foreach argument
    elif args.opt == "sf-sgd":
        sf_warmup_steps = 0 if args.scheduler != "none" else args.warmup_steps
        opt = SGDScheduleFree(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_steps=sf_warmup_steps,
            r=args.schedulefree_r,
            weight_lr_power=args.weight_lr_power,
        )  # without foreach argument
    elif args.opt == "amuse":
        sf_warmup_steps = 0 if args.scheduler != "none" else args.warmup_steps
        base_model = distributed_backend.get_raw_model(model)
        embed_params = [
            p
            for n, p in base_model.named_parameters()
            if ("embed" in n or "wte" in n or "wpe" in n)
            and not (hasattr(base_model, "lm_head") and p is base_model.lm_head.weight)
        ]
        scalar_params = [p for p in base_model.parameters() if p.ndim < 2]
        head_params = [base_model.lm_head.weight] if hasattr(base_model, "lm_head") else []
        assigned_params = embed_params + scalar_params + head_params
        assigned_param_ids = {id(p) for p in assigned_params}
        hidden_matrix_params = [
            p for p in base_model.parameters() if p.ndim >= 2 and id(p) not in assigned_param_ids
        ]

        adam_groups = dict(
            params=assigned_params,
            lr=args.lr,
            beta2=args.beta2,
            eps=1e-10,
            use_muon=False,
            weight_decay=args.weight_decay,
        )
        muon_group = dict(
            params=hidden_matrix_params,
            lr=args.lr,
            momentum=args.momentum,
            use_muon=True,
            weight_decay=args.weight_decay,
        )
        opt = AMUSE(
            param_groups=[adam_groups, muon_group],
            weight_decay_at_y=args.weight_decay_at_y,
            beta1=args.beta1,
            warmup_steps=sf_warmup_steps,
            rho=args.rho,
            r=args.schedulefree_r,
            weight_lr_power=args.weight_lr_power,
        )
    elif args.opt == "signsgd":
        opt = Signum(
            group_specs,
            lr=args.lr,
            momentum=0.0,  # always use zero momentum because its signSGD
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            sign_update=True,
        )
    elif args.opt == "signum":
        opt = Signum(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            dampening=args.dampening,
            nesterov=args.nesterov,
            sign_update=True,
        )
    elif args.opt == "prodigy":
        opt = Prodigy(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.weight_decay,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
            fsdp_in_use=args.prodigy_fsdp_in_use,
        )
    elif args.opt == "sophiag":
        opt = SophiaG(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            rho=args.sophia_rho,
        )
    elif args.opt == "adopt":
        opt = ADOPT(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.adopt_eps,  # 1e-6
            weight_decay=args.weight_decay,
            decouple=args.adopt_decouple,
        )
    elif args.opt == "mars":
        opt = MARS(
            group_specs,
            lr=args.mars_lr,
            betas=(args.mars_beta1, args.mars_beta2),
            weight_decay=args.weight_decay,
            amsgrad=False,
            gamma=args.mars_vr_gamma,
            is_approx=args.mars_is_approx,
            mars_type=args.mars_type,
            optimize_1d=False,  # we set in order to optimize 1D parameters with AdamW
            lr_1d=args.lr,  # AdamW's lr when optimize_1d=False
            betas_1d=(args.beta1, args.beta2),  # AdamW's betas when optimize_1d=False
            weight_decay_1d=0.1,  # AdamW's weight decay
        )
    elif args.opt == "adafactor":
        opt = Adafactor(
            group_specs,
            lr=args.lr,
            decay_rate=args.adafactor_decay_rate,
            beta1=args.beta1,
            clip_threshold=1.0,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "lamb":
        opt = Lamb(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            adam=False,
            bias_correction=args.lamb_use_bias_correction,
        )
    elif args.opt == "scion":
        scion_param_groups = scion_partitions(group_specs, model, args)
        scion_params_cnt = sum(
            p.numel() for group in scion_param_groups for p in group["params"]
        )
        print(f"Optimized parameters: {scion_params_cnt}")
        opt = Scion(
            scion_param_groups,
            lr=args.lr,
            momentum=args.momentum,
        )
    elif args.opt == "scion-light":
        scion_param_groups = scion_partitions(group_specs, model, args)
        scion_params_cnt = sum(
            p.numel() for group in scion_param_groups for p in group["params"]
        )
        print(f"Optimized parameters: {scion_params_cnt}")
        opt = ScionLight(
            scion_param_groups,
            lr=args.lr,
            momentum=args.momentum,
        )
    elif args.opt == "muon-pytorch":
        opt = torch.optim.Muon(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            ns_steps=args.muon_ns_steps,
            ns_coefficients=(
                3.4445,
                -4.775,
                2.0315,
            ),  # someone might try to change it later
            eps=1e-7,  # muon pytorch uses smaller eps
            adjust_lr_fn=None,  # to make the orthogonalized update have a consistent RMS across rectangular matrices
        )
    else:
        opt = torch.optim.SGD(
            group_specs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    print(f"\nOptimizer:\n{opt}")

    if args.scheduler != "none":
        assert (
            args.warmup_steps < args.iterations
        ), "Warmup steps must be < iterations."  # from schedules-and-scaling
        if args.scheduler in ["cos", "linear"]:
            # initial lr is args.lr / div_factor
            # final lr is initial_lr/final_div_factor = args.lr / div_factor / final_div_factor
            scheduler = (
                torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=opt,
                    max_lr=[
                        group.get("lr", args.lr) for group in group_specs
                    ],  # it was args.lr
                    total_steps=args.iterations,
                    pct_start=args.warmup_steps
                    / args.iterations,  # it was args.warmup_percent
                    anneal_strategy=args.scheduler,
                    cycle_momentum=False,
                    div_factor=1e2,
                    final_div_factor=args.final_div_factor,
                )
                if args.opt != "muon"
                else CombinedScheduler(opt, args)
            )
        elif args.scheduler == "linear_zero":
            def linear_warmup_decay_to_zero(step):
                t = step / args.iterations
                return max(0.0, 1.0 - t)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, linear_warmup_decay_to_zero)

        elif args.scheduler == "cos_inf":
            lambda_schedule = cos_inf_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                n_inf=args.cos_inf_steps,
                div_factor=1e2,
                final_div_factor=0.1,
            )
            scheduler = (
                torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
                if args.opt != "muon"
                else CombinedScheduler(opt, args)
            )
        elif args.scheduler == "wsd":
            lambda_schedule = wsd_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                fract_decay=args.wsd_fract_decay,
                init_div_factor=1e2,
                final_lr_factor=args.wsd_final_lr_scale,  # should be 0 here
                decay_type=args.decay_type,
            )
            scheduler = (
                torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
                if args.opt != "muon"
                else CombinedScheduler(opt, args)
            )
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None
        if args.opt == "d-muon" and args.warmup_steps > 0:
            scheduler = DMuonWarmupScheduler(opt, args.warmup_steps)


    latest_ckpt_dir = exp_dir / "ckpts" / "latest"
    latest_ckpt_path = latest_ckpt_dir / "main.pt"
    if latest_ckpt_path.exists():
        if args.resume_from is None:
            if not args.auto_resume:
                next_exp_name, next_exp_dir = next_fresh_experiment_dir(args, exp_name)
                if distributed_backend.is_master_process():
                    print(
                        f"[Fresh Start] checkpoint already exists in {exp_dir}; "
                        f"switching to {next_exp_dir}"
                    )
                exp_name, exp_dir = next_exp_name, next_exp_dir
                latest_ckpt_dir = exp_dir / "ckpts" / "latest"
                latest_ckpt_path = latest_ckpt_dir / "main.pt"
            else:
                args.resume_from = str(latest_ckpt_dir)
                if distributed_backend.is_master_process():
                    print(f"[Auto Resume] found checkpoint at {latest_ckpt_path}")
                    print(f"[Auto Resume] resuming from {args.resume_from}")
    elif distributed_backend.is_master_process():
        exp_dir.mkdir(parents=True, exist_ok=True)

    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
    )

    stats["args"] = vars(args)
    if distributed_backend.is_master_process():
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


def get_data_readers(args, verbose=True):
    data_srcs = get_dataset(args)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram,
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        keep_in_ram=args.data_in_ram,
    )

    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")

    return {
        "train": train_reader,
        "val": val_reader,
    }


def get_exp_name(
    args,
    parser,
    distributed_backend,
    key_args=["model", "dataset", "opt"],
    ignore_args=[
        "eval_interval",
        "full_eval_at",
        "distributed_backend",
        "latest_ckpt_interval",
        "permanent_ckpt_interval",
        "datasets_dir",
        "wandb",
        "wandb_project",
        "wandb_entity",
        "batch_size",
        "acc_steps",
        "results_base_folder",
        "run_prefix",
        "wandb_run_prefix",
        "seed",
        "device",
        "adema_beta3_warmup",
        "adema_alpha_warmup",
        "plot_router_logits",
        "weight_average",
        # "wa_interval",
        # "wa_horizon",
        "wa_dtype",
        "wa_use_temp_dir",
        "wa_sweep_horizon",
        # "max_num_wa_sweeps",
        "exponential_weight_average",
        # "ewa_interval",
        # "ewa_decay",
        # "ewa_after_warmup",
        "moe",
        "log_interval",
        "log_parameter_norms",
        "log_dynamics",
        "dynamics_logger_cfg",
        "experiment_name",
    ],
):
    # Set the custom exp name if needed
    if args.experiment_name is not None:
        return args.experiment_name

    # Get the default values
    defaults = vars(parser.parse_args([]))

    # rank = distributed_backend.rank # decided to remove rank from the exp name

    # Generate the prefix with key arguments
    prefix_parts = []
    for key in key_args:
        if hasattr(args, key):
            value = getattr(args, key)
            if key == "model":
                if getattr(args, "moe", False):
                    value = f"moe_{value}"
                if getattr(args, "weight_average", False):
                    value = f"{value}_WA"
                if getattr(args, "exponential_weight_average", False):
                    value = f"{value}_EWA"
            prefix_parts.append(f"{key}-{value}")

    prefix = "_".join(prefix_parts)
    prefix = f"{args.batch_size}x{args.acc_steps}_" + prefix  # rank={rank}

    # Generate the rest of the string with non-default arguments
    non_default_parts = []
    for key, value in vars(args).items():
        if key in ignore_args:
            continue
        if key not in defaults:
            print(f"Warning: {key} not in defaults")
            continue
        if key not in key_args and value != defaults[key]:
            non_default_parts.append(f"{key}-{value}")

    non_default_string = "_".join(non_default_parts)

    if args.run_prefix is not None:
        prefix = args.run_prefix + "_" + prefix

    # Combine prefix and non-default string
    if non_default_string:
        return f"{prefix}__{non_default_string}"
    else:
        return prefix


def resolve_experiment_dir(args, exp_name):
    exp_name = shorten_experiment_name(exp_name)
    exp_dir = Path(args.results_base_folder) / exp_name

    if args.resume_from is not None or args.auto_resume:
        return exp_name, exp_dir

    if not exp_dir.exists():
        return exp_name, exp_dir

    return next_fresh_experiment_dir(args, exp_name)


def next_fresh_experiment_dir(args, exp_name):
    base_name = re.sub(r"__fresh\d+$", "", exp_name)
    exp_dir = Path(args.results_base_folder) / base_name

    suffix = 1
    while True:
        candidate_name = f"{base_name}__fresh{suffix}"
        candidate_dir = Path(args.results_base_folder) / candidate_name
        if not candidate_dir.exists():
            print(
                f"[Fresh Start] experiment dir {exp_dir} exists; "
                f"starting a new run in {candidate_dir}"
            )
            return candidate_name, candidate_dir
        suffix += 1


def shorten_experiment_name(exp_name, max_len=180):
    if len(exp_name) <= max_len:
        return exp_name

    digest = hashlib.sha1(exp_name.encode("utf-8")).hexdigest()[:10]
    keep = max_len - len("__") - len(digest)
    keep = max(keep, 32)
    return f"{exp_name[:keep]}__{digest}"


if __name__ == "__main__":
    args, parser = get_args()
    main(args, parser)
