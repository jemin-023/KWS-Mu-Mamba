import os
import sys
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.kws.data import SpeechCommandsDataset, NUM_CLASSES, IDX_TO_CLASS
from train.kws.model import MuMambaKWS

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            loss_sum += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    acc = correct / total if total > 0 else 0.0
    avg_loss = loss_sum / total if total > 0 else 0.0
    return acc, avg_loss

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(8)
    print(f"Training on device: {device} (threads: {torch.get_num_threads()})")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing datasets...")
    train_dataset = SpeechCommandsDataset(
        data_dir=args.data_dir,
        split="train",
        augment=True,
    )
    val_dataset = SpeechCommandsDataset(
        data_dir=args.data_dir,
        split="validation",
        augment=False,
    )
    test_dataset = SpeechCommandsDataset(
        data_dir=args.data_dir,
        split="test",
        augment=False,
    )

    print(f"Dataset sizes: Train (per-epoch) = {len(train_dataset):,}, "
          f"Val = {len(val_dataset):,}, Test = {len(test_dataset):,}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    # Initialize model S
    model = MuMambaKWS(
        in_dim=40,
        d_model=args.d_model,
        d_inner=args.d_inner,
        d_state=args.d_state,
        dt_rank=args.dt_rank,
        d_conv=args.d_conv,
        num_layers=args.num_layers,
        num_classes=NUM_CLASSES,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (target ~18.1k)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0.0
    best_ckpt_path = save_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        # Re-sample unknown and silence classes for the new epoch
        train_dataset.resample_epoch()
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total:.3f}"
            })

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total
        epoch_time = time.time() - start_time

        val_acc, val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d} ({epoch_time:.1f}s): "
              f"Train Acc = {train_acc*100:.2f}%, Loss = {train_loss:.4f} | "
              f"Val Acc = {val_acc*100:.2f}%, Loss = {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
                "stats": {
                    "mean": torch.as_tensor(train_dataset.mean).float(),
                    "std": torch.as_tensor(train_dataset.std).float(),
                },
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val Acc: {val_acc*100:.2f}%)")

    print("\n--- Training Complete ---")
    if best_ckpt_path.exists():
        print(f"Loading best checkpoint from {best_ckpt_path} for final Test evaluation...")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    test_acc, test_loss = evaluate(model, test_loader, device)
    print(f"Final Test Set Accuracy: {test_acc*100:.2f}%, Test Loss: {test_loss:.4f}")
    return test_acc

def main():
    parser = argparse.ArgumentParser(description="Train Streaming Mu-Mamba KWS model")
    parser.add_argument("--data-dir", type=str, default="data/speech_commands/v0.02")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Architecture hyperparameters
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-inner", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=8)
    parser.add_argument("--dt-rank", type=int, default=4)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
