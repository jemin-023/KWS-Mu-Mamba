import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.kws.data import SpeechCommandsDataset, NUM_CLASSES, collate_sequences
from train.kws.model import MuMambaKWS, MambaC1Block
from train.kws.train import (
    sequence_criterion,
    build_val_sequences,
    evaluate_val_sequences,
    evaluate_test_clips,
)

def quantize_per_row_ste(w: torch.Tensor) -> torch.Tensor:
    """
    Symmetric per-row int8 fake quantization with Straight-Through Estimator (STE).
    w: (out_features, in_features)
    """
    scale = torch.amax(torch.abs(w), dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    w_scaled = w / scale
    w_q = torch.clamp(torch.round(w_scaled), -127.0, 127.0)
    w_fake = (w_q - w_scaled).detach() + w_scaled
    return w_fake * scale

def quantize_conv_ste(w: torch.Tensor) -> torch.Tensor:
    """
    Symmetric per-channel int8 fake quantization for Conv1d depthwise weights.
    w: (C_out, 1, K)
    """
    scale = torch.amax(torch.abs(w), dim=(1, 2), keepdim=True).clamp(min=1e-8) / 127.0
    w_scaled = w / scale
    w_q = torch.clamp(torch.round(w_scaled), -127.0, 127.0)
    w_fake = (w_q - w_scaled).detach() + w_scaled
    return w_fake * scale

class QATLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fake = quantize_per_row_ste(self.weight)
        return F.linear(x, w_fake, self.bias)

class QATConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fake = quantize_conv_ste(self.weight)
        return self._conv_forward(x, w_fake, self.bias)

def convert_to_qat(model: MuMambaKWS) -> MuMambaKWS:
    """
    Replaces Linear and Conv1d layers with their QAT counterparts.
    Latent float weights and biases are preserved in-place.
    """
    def _swap_linear(linear: nn.Linear) -> QATLinear:
        qat_l = QATLinear(linear.in_features, linear.out_features, bias=linear.bias is not None)
        with torch.no_grad():
            qat_l.weight.copy_(linear.weight)
            if linear.bias is not None:
                qat_l.bias.copy_(linear.bias)
        return qat_l

    def _swap_conv(conv: nn.Conv1d) -> QATConv1d:
        qat_c = QATConv1d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding=conv.padding,
        )
        with torch.no_grad():
            qat_c.weight.copy_(conv.weight)
            if conv.bias is not None:
                qat_c.bias.copy_(conv.bias)
        return qat_c

    # Stem & Head
    model.stem = _swap_linear(model.stem)
    model.head = _swap_linear(model.head)

    # Layers
    for layer in model.layers:
        layer.in_proj = _swap_linear(layer.in_proj)
        layer.conv1d = _swap_conv(layer.conv1d)
        layer.x_proj = _swap_linear(layer.x_proj)
        layer.dt_proj = _swap_linear(layer.dt_proj)
        layer.out_proj = _swap_linear(layer.out_proj)

    return model

def compute_multiplier_and_shift(scale_factor: float, max_shift: int = 24) -> Tuple[int, int]:
    """
    Decomposes a float scale factor S into integer multiplier M and right-shift P
    such that S ~= M / (2^P), with M in [0, 32767].
    """
    if scale_factor <= 0.0:
        return 0, 0
    # Find shift such that M is near 16384 (14-bit)
    shift = 0
    while scale_factor < 0.5 and shift < max_shift:
        scale_factor *= 2.0
        shift += 1
    mult = int(round(scale_factor * 32768.0))
    shift += 15
    mult = max(0, min(32767, mult))
    return mult, shift

def run_qat(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(12)
    print(f"Running Quantization-Aware Fine-Tuning (STE W8A16) on: {device} (threads: {torch.get_num_threads()})")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint {ckpt_path} not found. Complete float training first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = ckpt.get("args", {})
    stats = ckpt.get("stats", None)

    model = MuMambaKWS(
        in_dim=40,
        d_model=args_dict.get("d_model", 32),
        d_inner=args_dict.get("d_inner", 64),
        d_state=args_dict.get("d_state", 8),
        dt_rank=args_dict.get("dt_rank", 4),
        d_conv=args_dict.get("d_conv", 4),
        num_layers=args_dict.get("num_layers", 2),
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    print("Evaluating float baseline before QAT...")
    test_ds = SpeechCommandsDataset(data_dir=args.data_dir, split="test", augment=False, mode="clip")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    float_acc = evaluate_test_clips(model, test_loader, device)
    print(f"Baseline Float Test Accuracy: {float_acc:.2f}%")

    # Convert model to QAT modules
    model = convert_to_qat(model)

    train_ds = SpeechCommandsDataset(data_dir=args.data_dir, split="train", augment=True, mode="sequence")
    val_ds = SpeechCommandsDataset(data_dir=args.data_dir, split="validation", augment=False, mode="clip")
    val_seq_items = build_val_sequences(val_ds, num_sequences=200, seed=1337)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_val_score = 0.0
    save_path = Path(args.save_dir) / "quant_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_ds.resample_epoch(max_samples_per_class=args.max_samples_per_class)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_sequences,
        )

        model.train()
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{args.epochs}")
        for features, targets in pbar:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model.forward_seq(features)
            loss = sequence_criterion(logits, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        mean_score, kw_acc, bg_acc = evaluate_val_sequences(model, val_seq_items, device)
        print(f"QAT Epoch {epoch:2d}: Val Score = {mean_score*100:.2f}% (KW: {kw_acc*100:.2f}%, BG: {bg_acc*100:.2f}%)")
        if mean_score > best_val_score:
            best_val_score = mean_score
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_score": mean_score,
                "args": args_dict,
                "stats": stats,
            }, save_path)
            print(f"  --> Saved QAT checkpoint to {save_path} (Val Score: {mean_score*100:.2f}%)")

    # Final test evaluation with fake-quantized model
    if save_path.exists():
        ckpt_q = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_q["model_state_dict"])
    qat_test_acc = evaluate_test_clips(model, test_loader, device)
    drop = float_acc - qat_test_acc
    print(f"\nFinal QAT Test Accuracy: {qat_test_acc:.2f}% (drop from float: {drop:+.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="QAT Fine-tuning for Mu-Mamba")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--data-dir", type=str, default="data/speech_commands/v0.02")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples-per-class", type=int, default=200)
    args = parser.parse_args()
    run_qat(args)

if __name__ == "__main__":
    main()
