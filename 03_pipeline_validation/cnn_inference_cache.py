"""
Precompute CNN predictions (class, confidence) for every image in
simulation_fov_data.csv, for every trained backbone. Simulating GSS+CNN
across 71 FOV x 15 backbones x multiple p_min values would otherwise re-run
inference on the same images repeatedly -- caching once up front makes the
whole sweep fast.

Handles two checkpoint formats:
  - 14 model_zoo.py models: output/<name>_balanced/<name>_best.pt
  - yolo26n: output/yolo26n_balanced/weights/best.pt (Ultralytics format)

Output: one parquet file per model at
  output/cnn_cache/<model_name>_predictions.parquet
columns: dataset, fov_id, pos, abs_filepath, pred_class, confidence
(confidence = softmax probability of the predicted class)

Usage:
    python cnn_inference_cache.py                # all 15 models
    python cnn_inference_cache.py --model resnet18
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "02_backbone_training"))
from model_zoo import get_model, MODEL_REGISTRY  # noqa: E402

DATA_PATH = Path(__file__).parent / "output" / "simulation_fov_data.csv"
CHECKPOINT_DIR = Path(__file__).parent.parent / "02_backbone_training" / "output"
CACHE_DIR = Path(__file__).parent / "output" / "cnn_cache"
CLASS_NAMES = ["Blur", "Medium", "Focused"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ALL_MODELS = list(MODEL_REGISTRY.keys()) + ["yolo26n"]


class InferenceDataset(Dataset):
    def __init__(self, paths, input_size=224):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), idx


def run_model_zoo_inference(model_name, image_paths, device, batch_size=64):
    ckpt_path = CHECKPOINT_DIR / f"{model_name}_balanced" / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = get_model(model_name, num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = InferenceDataset(image_paths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    pred_classes = [None] * len(image_paths)
    confidences = [None] * len(image_paths)

    with torch.no_grad():
        for imgs, idxs in tqdm(loader, desc=f"[{model_name}] inference", leave=False):
            imgs = imgs.to(device)
            out = model(imgs)
            probs = torch.softmax(out, dim=1)
            conf, pred = probs.max(dim=1)
            for i, idx in enumerate(idxs.numpy()):
                pred_classes[idx] = CLASS_NAMES[pred[i].item()]
                confidences[idx] = conf[i].item()

    return pred_classes, confidences


def run_yolo26n_inference(image_paths, device):
    from ultralytics import YOLO

    ckpt_path = CHECKPOINT_DIR / "yolo26n_balanced" / "weights" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = YOLO(str(ckpt_path))
    pred_classes, confidences = [], []

    # Ultralytics batches internally; predict() accepts a list of paths directly.
    batch_size = 64
    for i in tqdm(range(0, len(image_paths), batch_size), desc="[yolo26n] inference", leave=False):
        batch = list(image_paths[i:i + batch_size])
        results = model.predict(batch, verbose=False, device=0 if device.type == "cuda" else "cpu")
        for r in results:
            top1 = r.probs.top1
            conf = r.probs.top1conf.item()
            pred_classes.append(r.names[top1])
            confidences.append(conf)

    return pred_classes, confidences


def cache_one_model(model_name, df, device):
    print(f"\n=== {model_name} ===")
    image_paths = df["abs_filepath"].values

    if model_name == "yolo26n":
        pred_classes, confidences = run_yolo26n_inference(image_paths, device)
    else:
        pred_classes, confidences = run_model_zoo_inference(model_name, image_paths, device)

    out_df = df[["dataset", "fov_id", "pos", "abs_filepath"]].copy()
    out_df["pred_class"] = pred_classes
    out_df["confidence"] = confidences

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{model_name}_predictions.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")

    pred_dist = pd.Series(pred_classes).value_counts(normalize=True) * 100
    print("Prediction distribution:", pred_dist.round(1).to_dict())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Single model to cache (default: all 15)")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    df["fov_id"] = df["fov_id"].astype(str)
    print(f"Loaded {len(df)} images across {df.groupby('dataset')['fov_id'].nunique().sum()} FOV.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models_to_run = [args.model] if args.model else ALL_MODELS

    for model_name in models_to_run:
        cache_path = CACHE_DIR / f"{model_name}_predictions.parquet"
        if cache_path.exists():
            print(f"\n=== {model_name} === already cached, skipping (delete file to re-run)")
            continue
        try:
            cache_one_model(model_name, df, device)
        except Exception as e:
            print(f"!! {model_name} FAILED: {e}")


if __name__ == "__main__":
    main()
