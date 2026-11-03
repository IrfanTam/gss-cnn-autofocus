"""
Generic training loop for any backbone in model_zoo.py.

Enforces the rules pinned in PROJECT.md:
  - class-weighted cross-entropy (imbalance: Focused ~3.3%, Medium ~2.8%, Blur ~93.8%)
  - per-class precision/recall/F1 reported, not just overall accuracy
    (accuracy alone is misleading here since Blur dominates)

Usage:
    python train.py --model focusnet_nano
    python train.py --model focusnet_micro --epochs 30
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

from model_zoo import get_model, count_params

DATA_DIR = Path(__file__).parent.parent / "01_data_preparation" / "output"
OUTPUT_DIR = Path(__file__).parent / "output"
CACHE_DIR = Path(__file__).parent / "resize_cache"
CLASS_NAMES = ["Blur", "Medium", "Focused"]  # alphabetical-independent, fixed order for labels

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# How much to subsample the Blur class in the TRAINING split only (val/test
# are left untouched so evaluation still reflects real-world class balance).
# Blur is downsampled to this many times the size of the next-largest
# minority class (Medium) -- see PROJECT.md decision log.
BLUR_SUBSAMPLE_RATIO = 3


class FocusDataset(Dataset):
    """
    Loads images through an on-disk resized-JPEG cache: source PNGs are
    960x760 and decoding+resizing them from scratch every epoch was the
    main training bottleneck (~25-30ms/image, see PROJECT.md). The first
    epoch populates the cache; every subsequent epoch (and every other
    model trained afterwards) reads the much smaller pre-resized JPEG
    instead, which is 5-10x faster to decode.
    """

    def __init__(self, df, transform, input_size=224):
        self.paths = df["abs_filepath"].values
        self.labels = df["label_idx"].values
        self.transform = transform
        self.input_size = input_size
        CACHE_DIR.mkdir(exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def _cache_path(self, path):
        # stable hash of the absolute path -> flat cache filename
        digest = hashlib.md5(path.encode()).hexdigest()
        return CACHE_DIR / f"{digest}_{self.input_size}.jpg"

    def __getitem__(self, idx):
        src_path = self.paths[idx]
        cache_path = self._cache_path(src_path)

        if cache_path.exists():
            img = Image.open(cache_path).convert("RGB")
        else:
            img = Image.open(src_path).convert("RGB")
            img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
            img.save(cache_path, "JPEG", quality=92)

        img = self.transform(img)
        return img, self.labels[idx]


def get_transforms(input_size=224):
    # Resize is NOT here -- FocusDataset's on-disk cache already stores
    # images pre-resized to input_size, so re-resizing per epoch would be
    # redundant work. If input_size changes, the cache key includes it,
    # so a new cache is populated automatically rather than serving stale
    # sizes.
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def subsample_blur(train_df, ratio=BLUR_SUBSAMPLE_RATIO, random_state=42):
    """
    Downsample the Blur class in the training split to `ratio` times the
    size of the next-largest minority class. Val/test are never touched --
    only called on train_df. See PROJECT.md decision log for rationale:
    this both speeds up training (fewer images/epoch) and reduces how
    aggressively the loss weights need to compensate for imbalance.
    """
    counts = train_df["focus_class"].value_counts()
    minority_max = counts.drop("Blur").max()
    target_blur_n = int(minority_max * ratio)

    blur_df = train_df[train_df["focus_class"] == "Blur"]
    other_df = train_df[train_df["focus_class"] != "Blur"]

    if len(blur_df) > target_blur_n:
        blur_df = blur_df.sample(target_blur_n, random_state=random_state)

    result = pd.concat([other_df, blur_df], ignore_index=True).sample(frac=1, random_state=random_state)
    print(f"Blur subsampled: {len(train_df[train_df['focus_class']=='Blur'])} -> {len(blur_df)} "
          f"({ratio}x the next-largest class, {minority_max})")
    return result.reset_index(drop=True)


def load_splits(balanced=False):
    """
    balanced=False (default): original ~94% Blur proportions, with Blur
        subsampled to 3x the next-largest class in TRAIN only (val/test
        untouched) -- see subsample_blur().
    balanced=True: reads the *_balanced.csv files (see
        01_data_preparation/data_preparation.py --balanced), which are
        already ~1:1:1 in every split. subsample_blur() is skipped since
        there's no majority class left to downsample -- these files are a
        DIFFERENT dataset variant, not the same one pre-subsampled twice.
    """
    suffix = "_balanced" if balanced else ""
    train_df = pd.read_csv(DATA_DIR / f"train{suffix}.csv")
    val_df = pd.read_csv(DATA_DIR / f"val{suffix}.csv")
    test_df = pd.read_csv(DATA_DIR / f"test{suffix}.csv")

    if not balanced:
        train_df = subsample_blur(train_df)

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    for df in (train_df, val_df, test_df):
        df["label_idx"] = df["focus_class"].map(class_to_idx)

    return train_df, val_df, test_df


def compute_class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes)
    total = len(labels)
    weights = total / (n_classes * counts)
    return weights.tolist()


class TooSlowError(RuntimeError):
    """Raised when a model's training speed is far below what's normal for
    this pipeline (see PROJECT.md benchmarks: FocusNet-Nano ~11.7 it/s,
    ResNet-18 still comfortably sub-minute per epoch after the resize-cache
    fix). Something at seconds-per-iteration is not "a big model", it's
    stuck -- OOM swapping, GPU not actually being used, another process
    hogging resources, etc. -- and is not worth waiting out."""


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs,
                     min_it_per_sec=None, speed_check_batches=10):
    """
    min_it_per_sec: if set, only checked during the FIRST epoch (epoch==0),
    over the first `speed_check_batches` batches. If the measured rate is
    below this threshold, raises TooSlowError instead of continuing --
    letting run_all.py's subprocess model move on to the next model rather
    than sit for hours on something stuck.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d}/{epochs} [train]", leave=False)

    check_speed = min_it_per_sec is not None and epoch == 0
    batch_start = time.time() if check_speed else None

    for batch_idx, (imgs, labels) in enumerate(pbar):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        if check_speed and batch_idx + 1 == speed_check_batches:
            elapsed = time.time() - batch_start
            rate = speed_check_batches / elapsed
            if rate < min_it_per_sec:
                raise TooSlowError(
                    f"Training speed {rate:.2f} it/s over first {speed_check_batches} batches "
                    f"is below the {min_it_per_sec} it/s floor -- aborting this model as stuck "
                    f"rather than waiting it out (see PROJECT.md for normal-speed benchmarks)."
                )
            check_speed = False  # only need to check once

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="val"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"[{desc}]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        total_loss += criterion(out, labels).item()
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}", acc=f"{correct/total:.4f}")
    return total_loss / len(loader), correct / total


@torch.no_grad()
def predict(model, loader, device, desc="predict"):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc=f"[{desc}]", leave=False):
        imgs = imgs.to(device)
        out = model(imgs)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_training_curves(history, out_path, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(ep, history["val_loss"], label="Val", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(ep, history["val_acc"], label="Val", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} -- Training Curves")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(labels, preds, class_names, out_path, model_name, split_name):
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Count")

    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", ax=axes[1],
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Normalized (row-wise)")

    plt.suptitle(f"{model_name} -- Confusion Matrix ({split_name})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run(model_name, epochs=50, batch_size=64, lr=1e-4, weight_decay=1e-2,
        patience=10, input_size=224, pretrained=True, min_it_per_sec=None, balanced=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_name} | Device: {device} | Balanced dataset: {balanced}")

    train_df, val_df, test_df = load_splits(balanced=balanced)
    train_tf, eval_tf = get_transforms(input_size)

    train_ds = FocusDataset(train_df, train_tf, input_size=input_size)
    val_ds = FocusDataset(val_df, eval_tf, input_size=input_size)
    test_ds = FocusDataset(test_df, eval_tf, input_size=input_size)

    # 12 CPU cores available on this machine. train/val loaders keep
    # persistent workers alive for the whole training loop (6 workers each
    # = 12 processes, already at the core count) -- the test loader is only
    # used once at the very end, so it gets a small non-persistent pool
    # instead of spawning 10 more processes on top of the ones still alive
    # from train/val (this was causing WinError 1455 / paging file
    # exhaustion after a long training run).
    train_val_kw = dict(batch_size=batch_size, num_workers=6, pin_memory=True,
                         persistent_workers=True)
    test_kw = dict(batch_size=batch_size, num_workers=2, pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True, **train_val_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **train_val_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **test_kw)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    model = get_model(model_name, num_classes=len(CLASS_NAMES), pretrained=pretrained).to(device)
    n_params = count_params(model)
    print(f"Params: {n_params:,}")

    cw = compute_class_weights(train_df["label_idx"].values, len(CLASS_NAMES))
    print(f"Class weights ({CLASS_NAMES}): {[f'{w:.3f}' for w in cw]}")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    patience_count = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    out_dir = OUTPUT_DIR / (f"{model_name}_balanced" if balanced else model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch_log_path = out_dir / "epoch_log.csv"
    with open(epoch_log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,is_best,elapsed_sec\n")

    start_time = time.time()
    for epoch in range(epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs,
                                         min_it_per_sec=min_it_per_sec)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device, desc=f"Epoch {epoch+1:02d}/{epochs} [val]")
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d}/{epochs}  train_loss={t_loss:.4f} train_acc={t_acc:.4f}  "
              f"val_loss={v_loss:.4f} val_acc={v_acc:.4f}  lr={current_lr:.2e}")

        is_best = v_loss < best_val_loss
        if is_best:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_count = 0
        else:
            patience_count += 1

        # Append immediately, not buffered until the end -- this file stays
        # complete and readable even if training is interrupted mid-run.
        with open(epoch_log_path, "a") as f:
            f.write(f"{epoch+1},{t_loss:.6f},{t_acc:.6f},{v_loss:.6f},{v_acc:.6f},"
                    f"{current_lr:.6e},{int(is_best)},{time.time()-start_time:.1f}\n")

        if patience_count >= patience:
            print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f} at epoch {best_epoch})")
            break

    # Capture the LAST-epoch weights before overwriting model with the BEST
    # checkpoint -- both are saved to disk below, since which one is more
    # useful depends on downstream use (best = lowest val_loss, typically
    # what you want; last = final state, useful for diagnosing whether
    # training was still improving when it stopped).
    last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    last_epoch = epoch + 1

    train_time = time.time() - start_time
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    torch.save({"model_state_dict": last_state, "epoch": last_epoch, "history": history},
               out_dir / f"{model_name}_last.pt")
    print(f"Saved last-epoch checkpoint (epoch {last_epoch}) -> {out_dir / f'{model_name}_last.pt'}")

    plot_training_curves(history, out_dir / "training_curves.png", model_name)

    # Val confusion matrix uses the best-checkpoint model (per-sample preds --
    # evaluate() during the loop only tracked aggregate loss/acc, not preds).
    val_preds, val_labels = predict(model, val_loader, device, desc="val predict")
    plot_confusion_matrix(val_labels, val_preds, CLASS_NAMES,
                           out_dir / "confusion_matrix_val.png", model_name, "Validation")

    # Explicitly release train/val persistent workers before spawning the
    # test loader's own worker pool -- otherwise both stay alive at once,
    # which is what triggered WinError 1455 (paging file exhaustion).
    del train_loader, val_loader

    preds, labels = predict(model, test_loader, device, desc="test predict")
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4, output_dict=True)
    report_text = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    cm = confusion_matrix(labels, preds).tolist()
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")

    plot_confusion_matrix(labels, preds, CLASS_NAMES,
                           out_dir / "confusion_matrix_test.png", model_name, "Test")

    print("\n" + "=" * 60)
    print(f"TEST RESULTS -- {model_name}")
    print("=" * 60)
    print(report_text)
    print(f"Overall accuracy: {acc:.4f}  Macro F1: {f1_macro:.4f}")
    print(f"Training time: {train_time:.1f}s ({len(history['train_loss'])} epochs)")

    # Inference time (per-image, batch size 1, after warmup)
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
        n_runs = 50
        t0 = time.time()
        for _ in range(n_runs):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.time() - t0) / n_runs * 1000

    torch.save({"model_state_dict": model.state_dict(), "epoch": best_epoch, "history": history},
               out_dir / f"{model_name}_best.pt")
    print(f"Saved best checkpoint (epoch {best_epoch}, val_loss={best_val_loss:.4f}) -> "
          f"{out_dir / f'{model_name}_best.pt'}")

    summary = {
        "model_name": model_name,
        "num_params": n_params,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "model_size_mb": round((out_dir / f"{model_name}_best.pt").stat().st_size / 1e6, 2),
        "epochs_trained": len(history["train_loss"]),
        "train_time_sec": round(train_time, 1),
        "inference_ms": round(inference_ms, 3),
        "test_accuracy": round(float(acc), 4),
        "test_f1_macro": round(float(f1_macro), 4),
        "per_class": {
            cls: {k: round(float(v), 4) for k, v in report[cls].items()}
            for cls in CLASS_NAMES
        },
        "confusion_matrix": cm,
        "class_names_order": CLASS_NAMES,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"{model_name} -- Test Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report_text)

    print(f"\nSaved -> {out_dir}/")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name from model_zoo.MODEL_REGISTRY")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-pretrained", action="store_true",
                         help="Train from scratch instead of ImageNet init (irrelevant for custom models)")
    parser.add_argument("--min-it-per-sec", type=float, default=None,
                         help="Abort (raise TooSlowError, exit code 1) if the first 10 batches of "
                              "epoch 1 run below this rate. Normal range on this pipeline after the "
                              "resize-cache fix is ~1-12 it/s depending on model size (see PROJECT.md). "
                              "Something at e.g. 28 SECONDS/it is stuck, not just 'a big model'.")
    parser.add_argument("--balanced", action="store_true",
                         help="Use the 1:1:1 balanced dataset variant (train/val/test all ~6,405 "
                              "images total, see 01_data_preparation/data_preparation.py --balanced) "
                              "instead of the default ~94%% Blur / 3%% Medium / 3%% Focused split. "
                              "Results saved to output/<model>_balanced/ -- does not overwrite the "
                              "unbalanced run. See PROJECT.md for the evaluation-representativeness "
                              "trade-off this implies.")
    args = parser.parse_args()

    run(args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        patience=args.patience, pretrained=not args.no_pretrained,
        min_it_per_sec=args.min_it_per_sec, balanced=args.balanced)
