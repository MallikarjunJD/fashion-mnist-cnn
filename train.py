"""
Training script for FashionCNN on Fashion-MNIST.

Trains with AdamW + OneCycleLR, checkpoints only the best-validation-accuracy
model (so the deployed weights are never an accidentally-overfit last epoch),
and exports the final weights ready for the deploy/ apps to load.

Run directly:  python train.py
"""

import argparse
import torch
import torch.nn as nn

from data import get_dataloaders, CLASS_NAMES
from model import FashionCNN


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)                # forward pass
        loss = criterion(logits, yb)
        loss.backward()                    # backpropagation
        optimizer.step()                   # weight update
        scheduler.step()                   # advance LR schedule every batch
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total


def run(epochs: int = 15, batch_size: int = 128, lr: float = 1e-3,
        seed: int = 42, checkpoint_path: str = "fashion_cnn.pt"):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )

    best_acc = 0.0
    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)   # checkpoint best only

        print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} "
              f"| Val Loss {val_loss:.4f} Acc {val_acc:.4f}"
              f"{'  <- saved' if val_acc == best_acc else ''}")

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print(f"Best weights saved to: {checkpoint_path}")
    print(f"Classes: {CLASS_NAMES}")
    return model, best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FashionCNN on Fashion-MNIST")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="fashion_cnn.pt")
    args = parser.parse_args()

    run(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, checkpoint_path=args.checkpoint)