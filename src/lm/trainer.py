import copy
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import wandb
import yaml
import torch.distributed as dist
from logger.logger import DynamicsLogger
from optim.weight_averaging import (ExponentialWeightAverager,
                                    WeightAverager, eval_ewa,
                                    eval_wa)

from optim.utils import (eval, get_batch, get_parameter_norms, load_checkpoint,
                         load_worker_state, log_prodigy_lr, save_checkpoint,
                         save_worker_state, visualize_routing)


def _format_decay_suffix(decay: float) -> str:
    return str(decay).replace(".", "p")


def _reset_optimizer_lrs(opt, scheduler, fallback_lr):
    if scheduler is not None and hasattr(scheduler, "base_lrs"):
        base_lrs = list(scheduler.base_lrs)
    else:
        base_lrs = [fallback_lr] * len(opt.param_groups)

    if len(base_lrs) == 1 and len(opt.param_groups) > 1:
        base_lrs = base_lrs * len(opt.param_groups)

    for group, lr in zip(opt.param_groups, base_lrs):
        group["lr"] = lr
        if "initial_lr" in group:
            group["initial_lr"] = lr


def _dist_barrier():
    if not dist.is_initialized():
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def _sync_device(device):
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _dist_max_float(value, device):
    if not dist.is_initialized():
        return value
    tensor_device = torch.device(device) if "cuda" in str(device) else torch.device("cpu")
    t = torch.tensor(value, device=tensor_device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    not_compiled_model = model
    start_itr = 0
    log_iter_offset = 0

    if getattr(cfg, "resume_from", None) is not None:
        resume_mode = getattr(cfg, "resume_mode", "full")
        load_optimizer = (resume_mode == "full")
        load_scheduler = (resume_mode == "full")

        resumed_itr, ckpt_dir, scheduler_loaded = load_checkpoint(
            model=not_compiled_model,
            opt=opt,
            scheduler=scheduler,
            ckpt_path=cfg.resume_from,
            device=cfg.device,
            load_optimizer=load_optimizer,
            load_scheduler=load_scheduler,
        )

        if load_optimizer and not scheduler_loaded:
            _reset_optimizer_lrs(opt, scheduler, cfg.lr)

        if resume_mode == "full":
            rank = 0 if not dist.is_initialized() else dist.get_rank()
            worker_state_path = ckpt_dir / f"worker_{rank}.pt"
            if worker_state_path.exists():
                load_worker_state(ckpt_dir)

        start_itr = resumed_itr + 1
        if resume_mode != "full":
            start_itr = 0  # If not full resume, start from iteration 0 (model weights are loaded, but not optimizer/scheduler state)
            log_iter_offset = max(resumed_itr, 0)
        
        if distributed_backend.is_master_process():
            print(f"[Resume] loaded checkpoint from {cfg.resume_from}")
            print(f"[Resume] resume_mode = {resume_mode}")
            print(f"[Resume] starting from itr={start_itr}")
            if load_optimizer and not scheduler_loaded:
                print("[Resume] optimizer state restored, scheduler restarted from current args")
            if log_iter_offset:
                print(f"[Resume] wandb/log iter offset = {log_iter_offset}")

    if cfg.compile:
        print("Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    curr_iter = start_itr
    
    # if cfg.resume_from:
    #     # This is a full resume including the model weights, optimizer, state
    #     # dataloader state, random seed, etc. Not indended for fine tuning or
    #     # other scenarios where some of these should change.
    #     print(f"\nResuming Training From {cfg.resume_from}")
    #     ckpt_dir = Path(cfg.resume_from)
    #     curr_iter = load_checkpoint(
    #         model,
    #         opt,
    #         scheduler,
    #         ckpt_dir / "main.pt",
    #         cfg.device,
    #     )
    #     load_worker_state(ckpt_dir)
    # else:
    #     curr_iter = 0

    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )
    is_schedulefree = cfg.opt in [
        "sf-sgd",
        "sf-adamw",
        "amuse",
    ]
    if cfg.exponential_weight_average:
        if is_schedulefree:
            x_ema_decays = getattr(cfg, "x_ema_decays", None) or [cfg.ewa_decay]
            x_ema_decays = list(dict.fromkeys(float(decay) for decay in x_ema_decays))
            ewa_x_list = [
                (
                    decay,
                    ExponentialWeightAverager(
                        not_compiled_model,
                        interval=cfg.ewa_interval,
                        decay=decay,
                        warmup=cfg.warmup_steps if cfg.ewa_after_warmup else 0,
                        dtype={
                            "float32": torch.float32,
                            "float64": torch.float64,
                        }[cfg.wa_dtype],
                    ),
                )
                for decay in x_ema_decays
            ]
        else:
            ema_decays = getattr(cfg, "ema_decays", None) or [cfg.ewa_decay]
            ema_decays = list(dict.fromkeys(float(decay) for decay in ema_decays))
            ewa_list = [
                (
                    decay,
                    ExponentialWeightAverager(
                        not_compiled_model,
                        interval=cfg.ewa_interval,
                        decay=decay,
                        warmup=cfg.warmup_steps if cfg.ewa_after_warmup else 0,
                        dtype={
                            "float32": torch.float32,
                            "float64": torch.float64,
                        }[cfg.wa_dtype],
                    ),
                )
                for decay in ema_decays
            ]

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    substep = curr_iter * cfg.acc_steps
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    train_wall_start = time.perf_counter()
    grad_norms = []
    def logged_iter(local_iter):
        return local_iter + log_iter_offset
    model.train()

    while curr_iter <= cfg.iterations:
        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            eval_and_log(
                tokens,
                curr_iter,
                logged_iter(curr_iter),
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                full_eval=(curr_iter in cfg.full_eval_at),
            )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

            if cfg.exponential_weight_average:
                if is_schedulefree:
                    for decay, ewa_x in ewa_x_list:
                        eval_ewa(
                            curr_iter,
                            not_compiled_model,
                            ewa_x,
                            val_reader,
                            type_ctx,
                            distributed_backend,
                            cfg,
                            full_eval=(curr_iter in cfg.full_eval_at),
                            prefix=f"xema_d{_format_decay_suffix(decay)}",
                        )
                else:
                    use_multi_ema = getattr(cfg, "ema_decays", None) is not None
                    for decay, ewa in ewa_list:
                        eval_ewa(
                            curr_iter,
                            not_compiled_model,
                            ewa,
                            val_reader,
                            type_ctx,
                            distributed_backend,
                            cfg,
                            full_eval=(curr_iter in cfg.full_eval_at),
                            prefix=f"ema_d{_format_decay_suffix(decay)}" if use_multi_ema else "ewa",
                        )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        _sync_device(cfg.device)
        t_start = time.perf_counter()
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y, moe=cfg.moe)

            loss = outputs["loss"] / cfg.acc_steps
            loss.backward()
            substep += 1

        if cfg.grad_clip != 0.0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                grad_norms.append(torch.nn.utils.clip_grad_norm_(
                    model.module.parameters(), cfg.grad_clip
                ))
            else:
                grad_norms.append(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                ))

        if is_schedulefree:
            opt.train()
        (
            opt.step()
            if cfg.opt != "sophiag"
            else opt.step(bs=cfg.sophia_bs * cfg.sequence_length)
        )
        if scheduler is not None:
            scheduler.step()
        if cfg.opt == "sophiag":
            opt.zero_grad(set_to_none=True)
            if curr_iter % cfg.precondition_frequency == cfg.precondition_frequency - 1:
                sample_again = model(x, targets=y, get_logits=True)
                samp_dist = torch.distributions.Categorical(
                    logits=sample_again["logits"]
                )
                y_sample = samp_dist.sample()
                loss_sampled = torch.nn.functional.cross_entropy(
                    sample_again["logits"].view(-1, sample_again["logits"].size(-1)),
                    y_sample.view(-1),
                    ignore_index=-1,
                )
                (loss_sampled / cfg.acc_steps).backward()
                opt.update_hessian()
                opt.zero_grad(set_to_none=True)
                model.zero_grad()
        elif cfg.opt == "mars":
            opt.zero_grad(set_to_none=True)
            opt.update_last_grad()
        else:
            opt.zero_grad(set_to_none=True)

        if cfg.weight_average:
            weight_averager.step(
                not_compiled_model, distributed_backend.is_master_process()
            )
        if cfg.exponential_weight_average:
            if is_schedulefree:
                opt.eval()
                for _, ewa_x in ewa_x_list:
                    ewa_x.step(not_compiled_model, distributed_backend.is_master_process())
                opt.train()
            else:
                for _, ewa in ewa_list:
                    ewa.step(not_compiled_model, distributed_backend.is_master_process())

        _sync_device(cfg.device)
        dt = time.perf_counter() - t_start
        elapsed_wall = time.perf_counter() - train_wall_start
        dt = _dist_max_float(dt, cfg.device)
        elapsed_wall = _dist_max_float(elapsed_wall, cfg.device)

        curr_iter += 1

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps
            train_aux_losses = {
                f"train/{k}": v for k, v in outputs["aux_losses"].items()
            }

            current_lrs = [
                param_group.get("scheduled_lr", param_group["lr"])
                for param_group in opt.param_groups
            ]

            if cfg.opt == "prodigy":
                prodigy_efective_lrs = log_prodigy_lr(opt)

            print(
                f"Train: Iter={logged_iter(curr_iter)} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_wall={dt:.2e}s "
                f"lr={current_lrs[0]:.2e}"
            )
            if cfg.opt == "prodigy":
                print(f"effective_lr={prodigy_efective_lrs[0]:.2e}")

            if cfg.wandb:
                wandb_logs = {
                    "tokens": tokens,
                    "iter": logged_iter(curr_iter),
                    "train/loss": train_loss,
                    "train/perplexity": 2.71828**train_loss,
                    "lr": current_lrs[0],
                    "iter_dt": dt,
                    "max_grad_norm": max(grad_norms).item() if grad_norms else 0,
                    "mean_grad_norm": (
                        torch.tensor(grad_norms).mean().item() if grad_norms else 0
                    ),
                    **train_aux_losses,
                }

                if cfg.opt == "prodigy":
                    wandb_logs["effective_lr"] = prodigy_efective_lrs[0]

                if cfg.opt == "amuse":
                    wandb_logs["beta_t"] = opt.param_groups[0].get(
                        "beta1", getattr(opt, "beta1_init", 0.0)
                    )

                if cfg.log_parameter_norms:
                    raw_model = distributed_backend.get_raw_model(model)
                    model_norm = get_parameter_norms(raw_model, order=cfg.norm_order)
                    wandb_logs["model_norm"] = model_norm

                wandb.log(wandb_logs)
            grad_norms = []
    return stats


def eval_and_log(
    tokens,
    curr_iter,
    log_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
):
    _dist_barrier()

    if not distributed_backend.is_master_process():
        _dist_barrier()
        return

    model.eval()
    sf_opts = [
        "sf-sgd",
        "sf-adamw",
        "amuse",
    ]
    if cfg.opt in sf_opts:
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity, val_aux_losses, router_logits = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        moe=cfg.moe,
        get_router_logits=cfg.moe and cfg.plot_router_logits,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={log_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "tokens": tokens,
                "iter": log_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
                **val_aux_losses,
            }
        else:
            logs = {
                "tokens": tokens,
                "iter": log_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
                **val_aux_losses,
            }

        if cfg.moe and cfg.plot_router_logits:
            routing_logs = visualize_routing(router_logits, cfg)
            logs = {**logs, **routing_logs}

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(log_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})

    model.train()
    if cfg.opt in sf_opts:
        opt.train()
    _dist_barrier()
