import os
import sys
import copy
import time
import math
import random
import argparse
import hashlib
import resource
import subprocess
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.kws.data import (
    SpeechCommandsDataset,
    NUM_CLASSES,
    IDX_TO_CLASS,
    CLASS_TO_IDX,
    build_continuous_stream,
    collate_sequences,
)
from train.kws.frontend import extract_log_mel_c
from train.kws.model import MuMambaKWS

def sequence_criterion(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100):
    """
    Cross-entropy with ignore_index, averaged separately over
    keyword-window tokens and background tokens, then summed.
    logits: (B, T, 12)
    targets: (B, T)
    """
    logits_flat = logits.reshape(-1, 12)
    targets_flat = targets.reshape(-1)

    ce = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index, reduction="none")

    kw_mask = (targets_flat != ignore_index) & (targets_flat != CLASS_TO_IDX["silence"])
    bg_mask = (targets_flat == CLASS_TO_IDX["silence"])

    loss_kw = ce[kw_mask].mean() if kw_mask.any() else torch.tensor(0.0, device=logits.device)
    loss_bg = ce[bg_mask].mean() if bg_mask.any() else torch.tensor(0.0, device=logits.device)

    return loss_kw + loss_bg

class WeightEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
                else:
                    self.shadow[k].copy_(v)

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)

def build_val_sequences(val_dataset: SpeechCommandsDataset, num_sequences: int = 300, seed: int = 1337):
    print(f"Precomputing {num_sequences} fixed, seeded validation sequences (seed {seed})...")
    rng = random.Random(seed)
    val_items = []
    for _ in range(num_sequences):
        audio, targets, _ = build_continuous_stream(
            val_dataset,
            clip_info=None,
            is_training=False,
            rng=rng,
        )
        n_tokens = len(targets)
        mel = extract_log_mel_c(audio, target_frames=n_tokens)
        mel_float = mel.astype(np.float32)
        mel_norm = (mel_float - val_dataset.mean) / val_dataset.std
        val_items.append((torch.from_numpy(mel_norm).float(), torch.from_numpy(targets).long()))
    return val_items

def evaluate_val_sequences_detailed(model: nn.Module, val_items, device: torch.device) -> dict:
    model.eval()
    kw_correct, kw_total = 0, 0
    bg_correct, bg_total = 0, 0
    bg_before_correct, bg_before_total = 0, 0
    bg_after_correct, bg_after_total = 0, 0

    with torch.no_grad():
        for feats, targets in val_items:
            feats = feats.unsqueeze(0).to(device, non_blocking=(device.type == "cuda"))
            targets = targets.to(device, non_blocking=(device.type == "cuda"))
            logits = model.forward_seq(feats).squeeze(0) # (T, 12)
            preds = logits.argmax(dim=-1)

            kw_mask = (targets != -100) & (targets != CLASS_TO_IDX["silence"])
            bg_mask = (targets == CLASS_TO_IDX["silence"])

            if kw_mask.any():
                kw_correct += (preds[kw_mask] == targets[kw_mask]).sum().item()
                kw_total += kw_mask.sum().item()
            if bg_mask.any():
                bg_correct += (preds[bg_mask] == targets[bg_mask]).sum().item()
                bg_total += bg_mask.sum().item()

            ignored = (targets == -100).nonzero(as_tuple=True)[0]
            first_ignored = ignored[0].item() if ignored.numel() > 0 else targets.numel()
            tok_pos = torch.arange(targets.numel(), device=device)
            before_mask = bg_mask & (tok_pos < first_ignored)
            after_mask = bg_mask & (tok_pos > first_ignored)
            if before_mask.any():
                bg_before_correct += (preds[before_mask] == targets[before_mask]).sum().item()
                bg_before_total += before_mask.sum().item()
            if after_mask.any():
                bg_after_correct += (preds[after_mask] == targets[after_mask]).sum().item()
                bg_after_total += after_mask.sum().item()

    return {
        "kw_acc": kw_correct / max(1, kw_total),
        "bg_acc": bg_correct / max(1, bg_total),
        "bg_acc_before_first_clip": bg_before_correct / max(1, bg_before_total),
        "bg_acc_after_first_clip": bg_after_correct / max(1, bg_after_total),
        "kw_total": kw_total,
        "bg_total": bg_total,
        "bg_before_total": bg_before_total,
        "bg_after_total": bg_after_total,
    }

def evaluate_val_sequences(model: nn.Module, val_items, device: torch.device):
    metrics = evaluate_val_sequences_detailed(model, val_items, device)
    kw_acc = metrics["kw_acc"]
    bg_acc = metrics["bg_acc"]
    mean_acc = (kw_acc + bg_acc) / 2.0
    return mean_acc, kw_acc, bg_acc

def evaluate_test_clips(model: nn.Module, test_loader: DataLoader, device: torch.device):
    """Evaluates clip accuracy using the 50-token zero-state last-token protocol."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))
            logits = model(features)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return (correct / max(1, total)) * 100.0

def get_lr(epoch: int, total_epochs: int, base_lr: float = 1e-3, warmup_epochs: int = 2) -> float:
    if epoch <= warmup_epochs:
        return base_lr * (epoch / warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.01 + 0.5 * (base_lr - base_lr * 0.01) * (1.0 + math.cos(math.pi * progress))

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    requested = getattr(args, "device", "auto")
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            print("Error: --device cuda requested but CUDA is not available (torch.cuda.is_available() is False).", file=sys.stderr)
            sys.exit(2)
        device = torch.device("cuda")
    else:  # cpu
        device = torch.device("cpu")
    use_cuda = device.type == "cuda"
    pin_memory = use_cuda
    non_blocking = use_cuda
    if device.type == "cpu":
        torch.set_num_threads(getattr(args, "num_threads", 4))
    print(f"Training on device: {device} (threads: {torch.get_num_threads()})")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU: {gpu_name} (total VRAM: {total_vram / (1024 ** 2):.0f} MiB)")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing datasets...")
    train_dataset = SpeechCommandsDataset(
        data_dir=args.data_dir,
        split="train",
        augment=True,
        mode="sequence",
    )
    val_dataset_clips = SpeechCommandsDataset(
        data_dir=args.data_dir,
        split="validation",
        augment=False,
        mode="clip",
    )
    test_dataset = SpeechCommandsDataset(
        data_dir=args.data_dir,
        split="test",
        augment=False,
        mode="clip",
    )

    val_seq_items = build_val_sequences(val_dataset_clips, num_sequences=300, seed=1337)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=args.workers,
                             pin_memory=pin_memory)

    # Initialize model S (D=32, L=2, N=8, R=4, K=4)
    model = MuMambaKWS(
        in_dim=40,
        d_model=args.d_model,
        d_inner=args.d_inner,
        d_state=args.d_state,
        dt_rank=args.dt_rank,
        d_conv=args.d_conv,
        num_layers=args.num_layers,
        num_classes=NUM_CLASSES,
        learned_a=getattr(args, "learned_a", False),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (Mu-Mamba size S)")

    # Optional warm start (--init-ckpt); without it training is from scratch
    init_ckpt = Path(args.init_ckpt) if args.init_ckpt else None
    if init_ckpt is not None:
        if not init_ckpt.exists():
            raise FileNotFoundError(f"--init-ckpt {init_ckpt} does not exist")
        print(f"Warm-starting from {init_ckpt}...")
        ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("Training FROM SCRATCH (no --init-ckpt).")
    git_state = _git_state()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = WeightEMA(model, decay=0.999)

    best_val_score = 0.0
    best_ckpt_path = save_dir / "best_model.pt"

    print(f"\nBeginning training for {args.epochs} epochs (lr={args.lr}, cosine decay, EMA)...")
    for epoch in range(1, args.epochs + 1):
        # Update learning rate
        cur_lr = get_lr(epoch, args.epochs, base_lr=args.lr, warmup_epochs=args.warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        train_dataset.resample_epoch(max_samples_per_class=args.max_samples_per_class)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_sequences,
            pin_memory=pin_memory,
        )

        model.train()
        total_loss = 0.0
        n_batches = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{args.epochs}")
        for features, targets in pbar:
            features = features.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            logits = model.forward_seq(features)
            loss = sequence_criterion(logits, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{cur_lr:.2e}"})

        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(1, n_batches)

        # Evaluate EMA model on validation sequences
        eval_model = copy.deepcopy(model)
        ema.apply_to(eval_model)
        metrics = evaluate_val_sequences_detailed(eval_model, val_seq_items, device)
        kw_acc = metrics["kw_acc"]
        bg_acc = metrics["bg_acc"]
        bg_before = metrics["bg_acc_before_first_clip"]
        bg_after = metrics["bg_acc_after_first_clip"]
        mean_score = (kw_acc + bg_acc) / 2.0

        print(f"Epoch {epoch:2d} ({epoch_time:.1f}s) | Loss: {avg_loss:.4f} | "
              f"Val Score: {mean_score*100:.2f}% (KW: {kw_acc*100:.2f}%, BG: {bg_acc*100:.2f}%, "
              f"BG-before: {bg_before*100:.2f}%, BG-after: {bg_after*100:.2f}%)")
        if use_cuda:
            peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Peak CUDA memory: {peak_mib:.0f} MiB")
        else:
            peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"Peak RSS: {peak_rss_mib:.0f} MiB")

        if mean_score > best_val_score:
            best_val_score = mean_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": eval_model.state_dict(),
                "val_score": mean_score,
                "kw_acc": kw_acc,
                "bg_acc": bg_acc,
                "bg_acc_before_first_clip": bg_before,
                "bg_acc_after_first_clip": bg_after,
                "args": vars(args),
                "init_ckpt": str(init_ckpt) if init_ckpt is not None else None,
                **git_state,  # git_commit, git_dirty, git_diff_sha256 (provenance only)
                "stats": {
                    "mean": torch.as_tensor(train_dataset.mean).float(),
                    "std": torch.as_tensor(train_dataset.std).float(),
                },
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Score: {mean_score*100:.2f}%)")

    print("\n--- Training Complete ---")
    if best_ckpt_path.exists():
        print(f"Loading best checkpoint from {best_ckpt_path} for final evaluation...")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    test_clip_acc = evaluate_test_clips(model, test_loader, device)
    print(f"Final Test Clip Accuracy: {test_clip_acc:.2f}%")
    return test_clip_acc

def _git_state() -> dict:
    """Checkpoint provenance: HEAD commit, whether the tree differs from it (untracked files count), and the sha256 of
    `git diff HEAD` (tracked changes only)."""
    root = Path(__file__).resolve().parent
    def run(*cmd):
        return subprocess.run(["git", *cmd], capture_output=True, check=True, cwd=root).stdout
    try:
        diff = run("diff", "HEAD")
        return {
            "git_commit": run("rev-parse", "HEAD").decode().strip(),
            "git_dirty": bool(run("status", "--porcelain").strip()),
            "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except Exception:
        return {"git_commit": "unknown", "git_dirty": None, "git_diff_sha256": None}

def main():
    parser = argparse.ArgumentParser(description="Retrain Streaming Mu-Mamba KWS model")
    parser.add_argument("--data-dir", type=str, default="data/speech_commands/v0.02")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Default 16 (reduced from 64: batch 64 with ~1000-token sequences exhausted RAM).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2,
                        help="Default 2 (reduced from 4: batch 64 with ~1000-token sequences exhausted RAM).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-class", type=int, default=160)
    parser.add_argument("--init-ckpt", type=str, default=None,
                        help="Checkpoint to warm-start from. Default: none (train from scratch).")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                        help="auto: cuda when torch.cuda.is_available() else cpu.")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="torch.set_num_threads on CPU.")

    # Architecture hyperparameters
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=8)
    parser.add_argument("--dt-rank", type=int, default=4)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learned-a", action="store_true",
                        help="Ablation: learn A = -exp(A_log) per (d, n) (init -(n+1)) and decay with exp(Delta*A) "
                             "instead of the frozen C1 powers. PyTorch only; there is no C engine for it. Note: AdamW weight "
                             "decay (--weight-decay, 1e-4) is applied to A_log like every other parameter.")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
