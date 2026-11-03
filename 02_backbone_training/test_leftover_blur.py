"""
Inference-only check: run a trained model against ALL Blur images it has
never seen (not in its train/val/test split at all), to verify whether
training on a class-balanced subset still generalizes to the real-world-
scale population of Blur images. See PROJECT.md for context.

Usage:
    python test_leftover_blur.py --model resnet18_balanced
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_zoo import get_model
from train import FocusDataset, get_transforms, CLASS_NAMES, OUTPUT_DIR

DATA_DIR = Path(__file__).parent.parent / "01_data_preparation" / "output"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                         help="Output subfolder name under output/, e.g. resnet18_balanced")
    parser.add_argument("--arch", default=None,
                         help="Architecture name for model_zoo.get_model (default: strip _balanced suffix)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    arch = args.arch or args.model.replace("_balanced", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    leftover_path = DATA_DIR / "leftover_blur_for_inference_test.csv"
    df = pd.read_csv(leftover_path)
    df["label_idx"] = CLASS_NAMES.index("Blur")  # all rows are Blur by construction
    print(f"Leftover Blur images (never seen by this model at all): {len(df):,}")

    _, eval_tf = get_transforms(224)
    ds = FocusDataset(df, eval_tf, input_size=224)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Checkpoints are saved as f"{arch}_best.pt" inside the output/<folder>/
    # dir -- the folder name may have a "_balanced" suffix but the file
    # itself does not (see train.py's out_dir/model_name split).
    ckpt_path = OUTPUT_DIR / args.model / f"{arch}_best.pt"
    if not ckpt_path.exists():
        ckpt_path = OUTPUT_DIR / args.model / f"{arch}.pt"  # older naming (pre-best/last split)
    print(f"Loading checkpoint: {ckpt_path}")

    model = get_model(arch, num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Inference on leftover Blur"):
            imgs = imgs.to(device)
            out = model(imgs)
            all_preds.extend(out.argmax(1).cpu().numpy())

    preds = np.array(all_preds)
    pred_names = [CLASS_NAMES[i] for i in preds]
    pred_counts = pd.Series(pred_names).value_counts()

    print("\n" + "=" * 60)
    print(f"PREDICTIONS on {len(df):,} leftover Blur images (ground truth: 100% Blur)")
    print("=" * 60)
    for cls in CLASS_NAMES:
        n = pred_counts.get(cls, 0)
        print(f"  Predicted {cls:8s}: {n:6,} ({n/len(df)*100:5.2f}%)")

    blur_idx = CLASS_NAMES.index("Blur")
    accuracy = (preds == blur_idx).mean()
    print(f"\nEffective recall on this held-out Blur population: {accuracy:.4f}")

    out_path = OUTPUT_DIR / args.model / "leftover_blur_inference.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_images": len(df),
            "predicted_counts": {cls: int(pred_counts.get(cls, 0)) for cls in CLASS_NAMES},
            "predicted_pct": {cls: round(float(pred_counts.get(cls, 0)) / len(df) * 100, 2) for cls in CLASS_NAMES},
            "blur_recall_on_leftover": round(float(accuracy), 4),
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
