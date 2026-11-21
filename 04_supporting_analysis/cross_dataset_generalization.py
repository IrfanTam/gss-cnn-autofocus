"""
Cross-dataset generalization test: leave-one-dataset-out, 5x, for the final
backbone (FocusNet-Nano) only. Each of the 5 source sub-datasets is a
distinct capture SESSION (different dates, see PROJECT.md) -- this tests
whether the classifier generalizes to an entirely unseen session, not just
unseen images within sessions it has already seen (that's what the
leftover-Blur test in Tahap 2 already covers).

For each held-out dataset:
  1. Train FocusNet-Nano from scratch on the OTHER 4 datasets, balanced 1:1:1
     (reusing the same balancing logic as 01_data_preparation).
  2. Evaluate on ALL images of the held-out dataset, UNBALANCED (real class
     proportions) -- this is the generalization signal we care about.

Usage:
    python cross_dataset_generalization.py                  # all 5 folds
    python cross_dataset_generalization.py --dataset Dataset_49FOV_90img
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "01_data_preparation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "02_backbone_training"))

from data_preparation import (  # noqa: E402
    load_all_datasets, exclude_anomalous_fov, label_p_i, balance_classes, DATASETS,
)
from model_zoo import get_model  # noqa: E402
from train import (  # noqa: E402
    FocusDataset, get_transforms, compute_class_weights,
    train_one_epoch, evaluate, predict, plot_confusion_matrix,
)

OUTPUT_DIR = Path(__file__).parent / "output"
CLASS_NAMES = ["Blur", "Medium", "Focused"]
ALL_DATASETS = list(DATASETS.keys())

EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-4
PATIENCE = 10


def load_full_labeled_data():
    """All 75,125 images, labeled, anomaly-excluded -- no balancing, no split yet."""
    df = load_all_datasets()
    df = exclude_anomalous_fov(df)
    df = label_p_i(df)
    df["fov_id"] = df["fov_id"].astype(str)
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    df["label_idx"] = df["focus_class"].map(class_to_idx)
    return df


def run_one_fold(full_df, held_out_dataset, model_arch="focusnet_nano"):
    print(f"\n{'='*70}\nHeld-out (test) dataset: {held_out_dataset}\n{'='*70}")

    train_pool = full_df[full_df["dataset"] != held_out_dataset].reset_index(drop=True)
    held_out_df = full_df[full_df["dataset"] == held_out_dataset].reset_index(drop=True)

    print(f"Train pool (4 datasets, pre-balance): {len(train_pool)} images")
    print(f"Held-out dataset (unbalanced, real proportions): {len(held_out_df)} images")
    print("Held-out class distribution:")
    print((held_out_df["focus_class"].value_counts(normalize=True) * 100).round(1))

    balanced_train = balance_classes(train_pool)
    print(f"Balanced train set: {len(balanced_train)} images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, eval_tf = get_transforms(224)

    train_ds = FocusDataset(balanced_train, train_tf, input_size=224)
    test_ds = FocusDataset(held_out_df, eval_tf, input_size=224)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=6, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = get_model(model_arch, num_classes=len(CLASS_NAMES), pretrained=True).to(device)
    cw = compute_class_weights(balanced_train["label_idx"].values, len(CLASS_NAMES))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # No held-out val split here -- train_pool is itself already excluded from
    # the test dataset, and the goal is cross-SESSION generalization, not
    # hyperparameter tuning. Use a small slice of the balanced train set as
    # a lightweight validation signal for early stopping only.
    n_val = max(1, int(len(balanced_train) * 0.1))
    val_df = balanced_train.sample(n_val, random_state=42)
    fit_df = balanced_train.drop(val_df.index)
    fit_ds = FocusDataset(fit_df, train_tf, input_size=224)
    val_ds = FocusDataset(val_df, eval_tf, input_size=224)
    fit_loader = DataLoader(fit_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=6, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        t_loss, t_acc = train_one_epoch(model, fit_loader, criterion, optimizer, device, epoch, EPOCHS)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device, desc=f"Epoch {epoch+1} [val]")
        scheduler.step(v_loss)
        print(f"Epoch {epoch+1:02d}/{EPOCHS}  train_loss={t_loss:.4f} train_acc={t_acc:.4f}  "
              f"val_loss={v_loss:.4f} val_acc={v_acc:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - start_time
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    preds, labels = predict(model, test_loader, device, desc="held-out test predict")
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4, output_dict=True)
    report_text = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")

    print(f"\n{report_text}")
    print(f"Held-out accuracy: {acc:.4f}  Macro F1: {f1_macro:.4f}  Train time: {train_time:.1f}s")

    out_dir = OUTPUT_DIR / "cross_dataset" / held_out_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(labels, preds, CLASS_NAMES,
                           out_dir / "confusion_matrix.png", model_arch, f"held-out: {held_out_dataset}")

    summary = {
        "held_out_dataset": held_out_dataset,
        "train_pool_datasets": [d for d in ALL_DATASETS if d != held_out_dataset],
        "n_train_balanced": len(balanced_train),
        "n_held_out": len(held_out_df),
        "held_out_class_dist_pct": (held_out_df["focus_class"].value_counts(normalize=True) * 100).round(2).to_dict(),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_macro), 4),
        "per_class": {cls: {k: round(float(v), 4) for k, v in report[cls].items()} for cls in CLASS_NAMES},
        "train_time_sec": round(train_time, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Single held-out dataset (default: all 5)")
    parser.add_argument("--arch", default="focusnet_nano")
    args = parser.parse_args()

    print("Loading full labeled dataset (75,125 images, anomaly-excluded)...")
    full_df = load_full_labeled_data()
    print(f"Loaded {len(full_df)} images across {full_df['dataset'].nunique()} datasets.")

    datasets_to_run = [args.dataset] if args.dataset else ALL_DATASETS

    all_summaries = []
    for held_out in datasets_to_run:
        summary = run_one_fold(full_df, held_out, model_arch=args.arch)
        all_summaries.append(summary)

    df_summary = pd.DataFrame([{
        "held_out_dataset": s["held_out_dataset"],
        "accuracy": s["accuracy"],
        "macro_f1": s["macro_f1"],
        "focused_recall": s["per_class"]["Focused"]["recall"],
        "medium_recall": s["per_class"]["Medium"]["recall"],
        "blur_recall": s["per_class"]["Blur"]["recall"],
        "n_held_out": s["n_held_out"],
    } for s in all_summaries])

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "cross_dataset_results.csv"
    df_summary.to_csv(out_path, index=False)
    print(f"\n{'='*70}\nSaved -> {out_path}")
    print(df_summary.to_string(index=False))

    if len(df_summary) > 1:
        print(f"\nMean accuracy: {df_summary['accuracy'].mean():.4f} +/- {df_summary['accuracy'].std():.4f}")
        print(f"Mean macro F1: {df_summary['macro_f1'].mean():.4f} +/- {df_summary['macro_f1'].std():.4f}")


if __name__ == "__main__":
    main()
