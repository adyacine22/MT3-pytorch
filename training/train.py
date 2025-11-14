"""Step-based MT3 training loop driven by the central project config."""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # pragma: no cover - older torch
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # type: ignore[assignment]

from configs import load_project_config
from configs.project_config import PROJECT_ROOT
from data.audio.augment import apply_augmentation
from data.audio import spectrogram
from data.datasets.loader import build_chunk_dataloader
from data.symbolic import vocabulary
from models.t5_pytorch import MT3Transformer
from models.t5_pytorch.config import load_t5_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MT3 with cached chunk datasets.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override manifest path (defaults to config cache chunk manifest).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (defaults to training.runtime.device).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override number of training steps.",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=None,
        help="Override validation frequency (steps).",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=None,
        help="Limit number of validation dataloader batches per evaluation (0 = full).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Override logging frequency (steps).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override checkpoint directory.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an existing checkpoint file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override dataloader batch size.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile even if enabled in config.",
    )
    return parser.parse_args()


def _log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}")


def _format_step_log(
    step: int,
    loss_name: str,
    loss_value: float,
    *,
    tokens: float | int | None,
    lr: float | None,
    elapsed: float | None,
    timings: Dict[str, float] | None,
    timing_order: Sequence[str] | None,
) -> str:
    parts = [f"step={step:06d}", f"{loss_name}={loss_value:.4f}"]
    if tokens is not None:
        parts.append(f"tokens={float(tokens):.0f}")
    if lr is not None:
        parts.append(f"lr={lr:.2e}")
    if elapsed is not None:
        parts.append(f"dt={elapsed:.2f}s")
    if timings:
        keys: List[str] = list(timing_order) if timing_order is not None else list(timings.keys())
        seen = set(keys)
        for key in timings.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
        label_map = {
            "dataloader": "data",
            "prepare_inputs": "prep",
            "forward": "fwd",
            "loss": "loss",
            "backward": "bwd",
            "optimization": "opt",
            "iteration": "iter",
        }
        time_parts: List[str] = []
        for key in keys:
            if key not in timings:
                continue
            label = "eval" if key == "evaluation" else label_map.get(key, key)
            suffix = "s"
            time_parts.append(f"{label}={timings[key]:.3f}{suffix}")
        if time_parts:
            parts.append(" ".join(time_parts))
    return " ".join(parts).strip()


def resolve_path(path: Path | None, default: str, root: Path) -> Path:
    base = Path(default) if path is None else Path(path)
    if not base.is_absolute():
        base = root / base
    return base


def compute_vocab_size() -> int:
    return max(r.max_id for r in vocabulary.EVENT_RANGES) + 1


def set_seed(seed: int, deterministic: bool, cudnn_benchmark: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark and not deterministic


def build_optimizer(model: torch.nn.Module, optim_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    opt_type = optim_cfg.get("optimizer", {}).get("type", "adamw").lower()
    params = model.parameters()
    if opt_type != "adamw":
        raise ValueError(f"Unsupported optimizer type: {opt_type}")
    opt_args = optim_cfg.get("optimizer", {})
    return torch.optim.AdamW(
        params,
        lr=float(opt_args.get("lr", 5e-4)),
        betas=tuple(opt_args.get("betas", (0.9, 0.999))),
        weight_decay=float(opt_args.get("weight_decay", 0.0)),
        eps=float(opt_args.get("eps", 1e-8)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: Dict[str, Any],
    loop_cfg: Dict[str, Any],
) -> LRScheduler | None:
    sched_type = scheduler_cfg.get("type", "").lower()
    if not sched_type:
        return None
    if sched_type != "cosine":
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
    warmup = int(scheduler_cfg.get("warmup_steps", 0))
    max_steps = int(loop_cfg.get("max_steps", 1))
    min_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = (step - warmup) / max(1, max_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return cast(LRScheduler, scheduler)


def maybe_compile(model: MT3Transformer, runtime_cfg: Dict[str, Any]) -> MT3Transformer:
    compile_cfg = runtime_cfg.get("compile", False)
    if isinstance(compile_cfg, bool):
        enabled = compile_cfg
        mode = "max-autotune"
        fullgraph = False
    else:
        enabled = bool(compile_cfg.get("enabled", False))
        mode = compile_cfg.get("mode", "max-autotune")
        fullgraph = bool(compile_cfg.get("fullgraph", False))
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")
    compiled = torch.compile(
        model,
        mode=mode,
        fullgraph=fullgraph,
    )
    return cast(MT3Transformer, compiled)


def _frame_lengths_from_metadata(
    metadata: Dict[str, List[Any]],
    sample_rate: int,
    hop_length: int,
    max_frames: int,
    chunk_frames: int,
    device: torch.device,
) -> torch.Tensor:
    hop_seconds = hop_length / float(sample_rate)
    lengths: List[int] = []
    starts: Iterable[float] = metadata.get("chunk_start_s", [])
    ends: Iterable[float] = metadata.get("chunk_end_s", [])
    for start, end in zip(starts, ends):
        duration = max(0.0, float(end) - float(start))
        if duration <= 0.0:
            frames = chunk_frames
        else:
            frames = math.ceil(duration / hop_seconds)
        frames = max(1, min(max_frames, frames))
        lengths.append(frames)
    if not lengths:
        lengths = [max_frames]
    return torch.tensor(lengths, dtype=torch.long, device=device)


def prepare_encoder_inputs(
    batch: Dict[str, Any],
    *,
    device: torch.device,
    sample_rate: int,
    hop_length: int,
    chunk_frames: int,
    augment_cfg: Dict[str, Any],
    log_shapes: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    waveforms = batch["waveform"].to(device=device, dtype=torch.float32, non_blocking=True)
    if augment_cfg.get("enabled", False):
        profiles = augment_cfg.get("profiles", ["none"])
        probability = float(augment_cfg.get("probability", 1.0))
        if probability >= 1.0:
            mask = torch.ones(waveforms.size(0), dtype=torch.bool, device=device)
        else:
            mask = torch.rand(waveforms.size(0), device=device) < probability
        if mask.any():
            augmented_subset = apply_augmentation(
                waveforms[mask],
                sample_rate=sample_rate,
                profiles=profiles,
            )
            waveforms[mask] = augmented_subset

    log_mel_batch = spectrogram.waveform_to_logmel(
        waveforms,
        sample_rate=sample_rate,
        device=device,
    )
    if log_mel_batch.ndim == 2:
        log_mel_batch = log_mel_batch.unsqueeze(0)
    encoder_inputs = log_mel_batch.permute(0, 2, 1).contiguous()
    if log_shapes:
        _log("ENCODER", f"encoder_inputs shape: {tuple(encoder_inputs.shape)}")
    frame_lengths = _frame_lengths_from_metadata(
        batch["metadata"],
        sample_rate=sample_rate,
        hop_length=hop_length,
        max_frames=encoder_inputs.size(1),
        chunk_frames=chunk_frames,
        device=device,
    )
    encoder_mask = (
        torch.arange(encoder_inputs.size(1), device=device)
        .unsqueeze(0)
        .lt(frame_lengths.unsqueeze(1))
    )
    return encoder_inputs, encoder_mask


def prepare_decoder_targets(
    batch: Dict[str, Any],
    device: torch.device,
    pad_id: int,
    *,
    log_shapes: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = batch["tokens"].to(device=device, dtype=torch.long, non_blocking=True)
    token_mask = batch["token_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    decoder_input = tokens[:, :-1]
    decoder_mask = token_mask[:, :-1]
    labels = tokens[:, 1:].clone()
    label_mask = token_mask[:, 1:]
    labels = labels.masked_fill(~label_mask, pad_id)
    if log_shapes:
        _log(
            "DECODER",
            f"decoder_input shape: {tuple(decoder_input.shape)}, labels shape: {tuple(labels.shape)}",
        )
    return decoder_input, decoder_mask, labels


def train_step(
    model: MT3Transformer,
    batch: Dict[str, Any],
    *,
    device: torch.device,
    sample_rate: int,
    hop_length: int,
    chunk_frames: int,
    augment_cfg: Dict[str, Any],
    pad_id: int,
    autocast_ctx,
    log_shapes: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    timings: Dict[str, float] = {}
    prep_start = time.time()
    encoder_inputs, encoder_mask = prepare_encoder_inputs(
        batch,
        device=device,
        sample_rate=sample_rate,
        hop_length=hop_length,
        chunk_frames=chunk_frames,
        augment_cfg=augment_cfg,
        log_shapes=log_shapes,
    )
    decoder_input, decoder_mask, labels = prepare_decoder_targets(
        batch,
        device,
        pad_id,
        log_shapes=log_shapes,
    )
    timings["prepare_inputs"] = time.time() - prep_start
    with autocast_ctx:
        forward_start = time.time()
        logits, _ = model(
            encoder_inputs,
            decoder_input,
            decoder_target_tokens=labels,
            encoder_padding_mask=encoder_mask,
            decoder_padding_mask=decoder_mask,
        )
        timings["forward"] = time.time() - forward_start
        loss_start = time.time()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=pad_id,
        )
        timings["loss"] = time.time() - loss_start
    token_count = decoder_mask.sum().detach()
    return loss, token_count, timings


def evaluate(
    model: MT3Transformer,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    sample_rate: int,
    hop_length: int,
    chunk_frames: int,
    pad_id: int,
    max_batches: int | None = None,
) -> Tuple[float, Dict[str, float], int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    timings: Dict[str, float] = defaultdict(float)
    eval_start = time.time()
    data_timer = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            fetch_done = time.time()
            timings["dataloader"] += fetch_done - data_timer
            prep_start = time.time()
            encoder_inputs, encoder_mask = prepare_encoder_inputs(
                batch,
                device=device,
                sample_rate=sample_rate,
                hop_length=hop_length,
                chunk_frames=chunk_frames,
                augment_cfg={"enabled": False, "profiles": ["none"], "probability": 0.0},
                log_shapes=batch_idx == 1,
            )
            decoder_input, decoder_mask, labels = prepare_decoder_targets(
                batch,
                device,
                pad_id,
                log_shapes=batch_idx == 1,
            )
            timings["prepare_inputs"] += time.time() - prep_start
            forward_start = time.time()
            logits, _ = model(
                encoder_inputs,
                decoder_input,
                decoder_target_tokens=labels,
                encoder_padding_mask=encoder_mask,
                decoder_padding_mask=decoder_mask,
            )
            timings["forward"] += time.time() - forward_start
            loss_start = time.time()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=pad_id,
            )
            timings["loss"] += time.time() - loss_start
            tokens = decoder_mask.sum().item()
            total_loss += loss.item() * max(1, tokens)
            total_tokens += max(1, tokens)
            data_timer = time.time()
            if max_batches is not None and batch_idx >= max_batches:
                break
    timings["evaluation"] = time.time() - eval_start
    model.train()
    if total_tokens == 0:
        return float("nan"), timings, total_tokens
    return total_loss / total_tokens, timings, total_tokens


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: MT3Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    scaler: GradScaler | None,
    best_metric: float,
) -> None:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
    }
    torch.save(payload, path)


def maybe_cleanup(directory: Path, pattern: str, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    excess = len(checkpoints) - keep
    for idx in range(max(0, excess)):
        checkpoints[idx].unlink(missing_ok=True)


def resume_if_needed(
    path: Path | None,
    *,
    model: MT3Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    scaler: GradScaler | None,
) -> Tuple[int, float]:
    if path is None or not path.exists():
        return 0, float("inf")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    best = state.get("best_metric", float("inf"))
    return int(state.get("step", 0)), float(best)


def main() -> None:
    args = parse_args()
    project_cfg = load_project_config()
    training_cfg = project_cfg["training"]
    dataloader_cfg = dict(training_cfg.get("dataloader", {}))
    loop_cfg = dict(training_cfg.get("loop", {}))
    runtime_cfg = dict(training_cfg.get("runtime", {}))
    precision_cfg = dict(training_cfg.get("precision", {}))
    checkpoint_cfg = dict(training_cfg.get("checkpointing", {}))
    scheduler_cfg = dict(training_cfg.get("scheduler", {}))
    optim_cfg = dict(training_cfg.get("optimization", {}))
    augment_cfg = dict(training_cfg.get("augmentation", {}))

    manifest_default = project_cfg["paths"]["cache"]["chunk_manifest"]
    manifest_path = resolve_path(args.manifest, manifest_default, PROJECT_ROOT)
    if args.disable_compile:
        runtime_cfg["compile"] = False
    device_str = args.device or runtime_cfg.get("device", "cuda")
    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    max_steps = args.max_steps or int(loop_cfg.get("max_steps", 0))
    val_every = args.val_every or int(loop_cfg.get("val_every_steps", 0))
    val_max_batches_cfg = (
        args.val_batches if args.val_batches is not None else int(loop_cfg.get("val_max_batches", 0))
    )
    val_max_batches = val_max_batches_cfg if val_max_batches_cfg > 0 else None
    log_every = args.log_every or int(loop_cfg.get("log_every_steps", 0))

    sample_rate = int(project_cfg["audio"]["io"]["sample_rate"])
    hop_length = int(project_cfg["audio"]["features"]["hop_length"])
    chunk_frames = int(project_cfg["audio"]["features"]["chunk_frames"])

    seed = int(runtime_cfg.get("seed", 42))
    deterministic = bool(runtime_cfg.get("deterministic", False))
    cudnn_benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
    set_seed(seed, deterministic, cudnn_benchmark)

    _log("CONFIG", f"Using manifest: {manifest_path}")
    _log("CONFIG", f"Training device: {device} (max_steps={max_steps}, val_every={val_every}, log_every={log_every})")

    train_batch_size = args.batch_size or int(dataloader_cfg.get("batch_size", 32))
    train_loader = build_chunk_dataloader(
        manifest_path,
        split="train",
        compute_log_mel_in_collate=False,
        collate_device=None,
        batch_size=train_batch_size,
    )
    val_loader = build_chunk_dataloader(
        manifest_path,
        split="validation",
        compute_log_mel_in_collate=False,
        collate_device=None,
        batch_size=train_batch_size,
    )
    if len(val_loader.dataset) == 0:  # type: ignore[arg-type]
        _log("DATA", "Validation split is empty; disabling validation.")
        val_loader = None
        val_every = 0
    train_size = len(getattr(train_loader, "dataset"))
    if val_loader is None:
        val_desc = "disabled"
    else:
        val_desc = str(len(getattr(val_loader, "dataset")))
    _log("DATA", f"Loaded dataloaders: train={train_size} examples, val={val_desc}")

    vocab_size = compute_vocab_size()
    model = MT3Transformer(load_t5_config(vocab_size=vocab_size))
    model.to(device)
    compile_cfg = runtime_cfg.get("compile", False)
    if isinstance(compile_cfg, bool):
        compile_enabled = compile_cfg
    else:
        compile_enabled = bool(compile_cfg.get("enabled", False))
    model = maybe_compile(model, runtime_cfg)
    _log("MODEL", f"Instantiated MT3Transformer (vocab={vocab_size}, compile={'enabled' if compile_enabled else 'disabled'})")

    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, scheduler_cfg, {"max_steps": max_steps})
    _log(
        "OPTIM",
        f"Optimizer=AdamW lr={optimizer.param_groups[0]['lr']:.2e}, "
        f"scheduler={'cosine' if scheduler is not None else 'none'}",
    )

    dtype_name = str(precision_cfg.get("dtype", "fp32")).lower()
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    amp_dtype = dtype_map.get(dtype_name, torch.float32)
    amp_enabled = device.type in {"cuda", "mps"} and amp_dtype != torch.float32
    use_grad_scaler = bool(
        precision_cfg.get("use_grad_scaler", amp_dtype == torch.float16 and device.type == "cuda")
    )
    scaler_device_type = device.type if device.type in {"cuda", "cpu"} else "cuda"
    scaler = GradScaler(scaler_device_type, enabled=amp_enabled and use_grad_scaler)

    checkpoint_dir = resolve_path(args.run_dir, checkpoint_cfg.get("save_dir", "checkpoints"), PROJECT_ROOT)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_step, best_metric = resume_if_needed(
        args.resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    if start_step:
        _log("CHECKPOINT", f"Resumed from step {start_step}")

    pad_id = -100
    grad_clip = float(optim_cfg.get("gradient_clipping", 0.0))
    save_every = int(checkpoint_cfg.get("save_every_steps", 0))
    keep_last = int(checkpoint_cfg.get("keep_last", 0))
    best_metric_name = checkpoint_cfg.get("metric", "val_loss")
    best_mode = checkpoint_cfg.get("mode", "min").lower()
    if best_mode == "max":
        best_metric = float("-inf")
    elif start_step == 0:
        best_metric = float("inf")

    train_iter = iter(train_loader)
    global_step = start_step
    last_log = time.time()
    running_loss = 0.0
    running_tokens = 0.0
    last_stage_times: Dict[str, float] = {}

    while global_step < max_steps:
        iter_start = time.time()
        data_fetch_start = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        dataloader_time = time.time() - data_fetch_start

        model.train()
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
            if amp_enabled
            else contextlib.nullcontext()
        )
        loss, token_count, stage_times = train_step(
            model,
            batch,
            device=device,
            sample_rate=sample_rate,
            hop_length=hop_length,
            chunk_frames=chunk_frames,
            augment_cfg=augment_cfg,
            pad_id=pad_id,
            autocast_ctx=autocast_ctx,
            log_shapes=(global_step == start_step),
        )
        stage_times["dataloader"] = dataloader_time

        backward_start = time.time()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        stage_times["backward"] = time.time() - backward_start

        optimize_start = time.time()
        if grad_clip > 0.0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        stage_times["optimization"] = time.time() - optimize_start

        global_step += 1
        running_loss += loss.item() * max(1, token_count.item())
        running_tokens += max(1, token_count.item())
        stage_times["iteration"] = time.time() - iter_start
        last_stage_times = stage_times

        if log_every and global_step % log_every == 0:
            avg_loss = running_loss / max(1.0, running_tokens)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - last_log
            train_log = _format_step_log(
                global_step,
                "loss",
                avg_loss,
                tokens=running_tokens,
                lr=lr,
                elapsed=elapsed,
                timings=last_stage_times,
                timing_order=(
                    "dataloader",
                    "prepare_inputs",
                    "forward",
                    "loss",
                    "backward",
                    "optimization",
                    "iteration",
                ),
            )
            _log("TRAIN", train_log)
            running_loss = 0.0
            running_tokens = 0.0
            last_log = time.time()

        if val_loader is not None and val_every and global_step % val_every == 0:
            val_loss, val_times, val_tokens = evaluate(
                model,
                val_loader,
                device=device,
                sample_rate=sample_rate,
                hop_length=hop_length,
                chunk_frames=chunk_frames,
                pad_id=pad_id,
                max_batches=val_max_batches,
            )
            val_elapsed = val_times.get("evaluation")
            val_log = _format_step_log(
                global_step,
                best_metric_name,
                val_loss,
                tokens=val_tokens,
                lr=optimizer.param_groups[0]["lr"],
                elapsed=val_elapsed,
                timings=val_times,
                timing_order=("evaluation", "dataloader", "prepare_inputs", "forward", "loss"),
            )
            _log("VAL", val_log)
            improved = (
                (best_mode == "min" and val_loss < best_metric)
                or (best_mode == "max" and val_loss > best_metric)
            )
            if improved:
                best_metric = val_loss
                best_path = checkpoint_dir / "best.pt"
                save_checkpoint(
                    best_path,
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_metric=best_metric,
                )
                _log("CHECKPOINT", f"Saved new best checkpoint to {best_path}")

        if save_every and global_step % save_every == 0:
            ckpt_path = checkpoint_dir / f"step_{global_step:06d}.pt"
            save_checkpoint(
                ckpt_path,
                step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_metric=best_metric,
            )
            maybe_cleanup(checkpoint_dir, "step_*.pt", keep_last)

    final_path = checkpoint_dir / f"final_step_{global_step:06d}.pt"
    save_checkpoint(
        final_path,
        step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_metric=best_metric,
    )
    _log("FINAL", f"Training finished, final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
