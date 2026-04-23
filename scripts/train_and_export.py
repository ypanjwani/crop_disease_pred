"""
scripts/train_and_export.py
─────────────────────────────────────────────────────────────────────────────
Training script for all three CNN architectures on the plant disease dataset.
Saves weights in the exact format expected by ModelRegistry.

Usage
─────
python scripts/train_and_export.py \
    --data_dir  /path/to/plant_disease_dataset \
    --output_dir backend/weights \
    --epochs     10 \
    --batch_size 32 \
    --lr         0.0003 \
    --model      all          # or: resnet18 | efficientnet_b0 | densenet121
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Allow importing from backend/ when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from models.architectures import ALL_ARCHITECTURES, NUM_CLASSES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("train")

# ── Image transforms ──────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_datasets(data_dir: Path, batch_size: int):
    train_ds = datasets.ImageFolder(data_dir / "train", TRAIN_TRANSFORM)
    val_ds   = datasets.ImageFolder(data_dir / "val",   EVAL_TRANSFORM)
    test_ds  = datasets.ImageFolder(data_dir / "test",  EVAL_TRANSFORM)

    log.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    log.info(f"Classes ({len(train_ds.classes)}): {train_ds.classes}")

    assert len(train_ds.classes) == NUM_CLASSES, (
        f"Dataset has {len(train_ds.classes)} classes but NUM_CLASSES={NUM_CLASSES}. "
        f"Update models/architectures.py → CLASS_NAMES."
    )

    make_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=min(8, os.cpu_count()), pin_memory=True,
    )
    return (
        make_loader(train_ds, True),
        make_loader(val_ds, False),
        make_loader(test_ds, False),
        train_ds.classes,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = correct = total = 0

    for step, (imgs, labels) in enumerate(loader, 1):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

        if step % 50 == 0:
            log.info(f"  [ep {epoch}  step {step}/{len(loader)}]  "
                     f"loss={running_loss/step:.4f}  acc={correct/total:.4f}")

    return running_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        running_loss += loss.item()
        correct      += (logits.argmax(1) == labels).sum().item()
        total        += labels.size(0)

    return running_loss / len(loader), correct / total


def train_model(key: str, data_dir: Path, output_dir: Path, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"\n{'='*60}")
    log.info(f"Training {key}  on {device}")
    log.info(f"{'='*60}")

    config   = ALL_ARCHITECTURES[key]()
    model    = config.model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader, val_loader, test_loader, _ = load_datasets(data_dir, args.batch_size)

    best_val_acc  = 0.0
    best_state    = None
    history       = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        log.info(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  "
            f"({elapsed:.0f}s)"
        )
        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc, vl_loss=vl_loss, vl_acc=vl_acc))

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log.info(f"  ↑ New best val_acc={best_val_acc:.4f} — checkpoint saved")

    # ── Final test evaluation ─────────────────────────────────────────────────
    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, test_loader, criterion, device)
    log.info(f"\n✅ {config.name}  test_acc={test_acc:.4f}  best_val_acc={best_val_acc:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / config.weight_key
    torch.save(
        {
            "model_key":       key,
            "model_name":      config.name,
            "num_classes":     NUM_CLASSES,
            "best_val_acc":    best_val_acc,
            "test_acc":        test_acc,
            "epochs_trained":  args.epochs,
            "history":         history,
            "model_state_dict": best_state,
        },
        save_path,
    )
    log.info(f"💾 Saved → {save_path}")
    return test_acc


def main():
    parser = argparse.ArgumentParser(description="Train CNN models for plant disease detection")
    parser.add_argument("--data_dir",   type=Path, required=True, help="Dataset root (must have train/val/test subfolders)")
    parser.add_argument("--output_dir", type=Path, default=Path("backend/weights"))
    parser.add_argument("--epochs",     type=int,  default=10)
    parser.add_argument("--batch_size", type=int,  default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--model",      type=str,  default="all",
                        choices=["all", "resnet18", "efficientnet_b0", "densenet121"])
    args = parser.parse_args()

    keys = list(ALL_ARCHITECTURES.keys()) if args.model == "all" else [args.model]

    results = {}
    for key in keys:
        results[key] = train_model(key, args.data_dir, args.output_dir, args)

    log.info("\n" + "="*60)
    log.info("TRAINING COMPLETE")
    log.info("="*60)
    for key, acc in results.items():
        log.info(f"  {key:<22s}  test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
