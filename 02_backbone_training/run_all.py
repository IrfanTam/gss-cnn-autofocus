"""
Train all 14 model_zoo backbones sequentially (YOLO26n is trained separately
via the Ultralytics pipeline -- not included here, see PROJECT.md).

Runs each model in its own subprocess so a crash (OOM, etc.) in one model
cannot take down the rest of the batch. Writes a running summary CSV/JSON
after every model, so progress survives even if the whole script is killed.

Usage:
    python run_all.py
    python run_all.py --models focusnet_nano focusnet_micro
    python run_all.py --skip resnet34 convnext_tiny swin_tiny   # e.g. too slow / OOM on this GPU
    python run_all.py --epochs 30 --batch-size 32
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from model_zoo import MODEL_REGISTRY

OUTPUT_DIR = Path(__file__).parent / "output"
RUN_LOG_PATH = OUTPUT_DIR / "run_all_log.json"


def load_run_log():
    if RUN_LOG_PATH.exists():
        with open(RUN_LOG_PATH) as f:
            return json.load(f)
    return {}


def save_run_log(log):
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(RUN_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def train_one(model_name, epochs, batch_size, lr, patience, min_it_per_sec):
    """Run train.py for a single model as a subprocess. Returns (status, message)."""
    cmd = [
        sys.executable, str(Path(__file__).parent / "train.py"),
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--patience", str(patience),
    ]
    if min_it_per_sec is not None:
        cmd += ["--min-it-per-sec", str(min_it_per_sec)]
    print(f"\n{'='*70}\nRunning: {' '.join(cmd)}\n{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        note = "too slow (aborted, see TooSlowError above)" if elapsed < 120 else "crashed/OOM"
        return "failed", f"exit code {result.returncode} after {elapsed:.0f}s ({note})"
    return "success", f"completed in {elapsed:.0f}s"


def collect_metrics(model_name):
    """Read back metrics.json for a model that succeeded, if present."""
    metrics_path = OUTPUT_DIR / model_name / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def build_comparison_table(log):
    rows = []
    for model_name, entry in log.items():
        if entry["status"] != "success":
            rows.append({"model": model_name, "status": entry["status"], "note": entry.get("message", "")})
            continue
        m = collect_metrics(model_name)
        if m is None:
            rows.append({"model": model_name, "status": "success (metrics.json missing)"})
            continue
        rows.append({
            "model": model_name,
            "status": "success",
            "params": m["num_params"],
            "model_size_mb": m["model_size_mb"],
            "inference_ms": m["inference_ms"],
            "epochs_trained": m["epochs_trained"],
            "train_time_sec": m["train_time_sec"],
            "test_accuracy": m["test_accuracy"],
            "test_f1_macro": m["test_f1_macro"],
            "focused_recall": m["per_class"].get("Focused", {}).get("recall"),
            "focused_precision": m["per_class"].get("Focused", {}).get("precision"),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None,
                         help="Subset of model names to train (default: all in MODEL_REGISTRY)")
    parser.add_argument("--skip", nargs="*", default=[],
                         help="Model names to skip (e.g. too slow/OOM on this GPU)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", action="store_true",
                         help="Skip models already marked 'success' in run_all_log.json")
    parser.add_argument("--min-it-per-sec", type=float, default=0.5,
                         help="Abort a model (and move to the next) if training runs below this "
                              "rate over the first 10 batches. Default 0.5 it/s (= 2 sec/it) is well "
                              "under every model's normal speed seen so far (FocusNet-Nano ~11.7 it/s, "
                              "ResNet-18 comfortably faster than that too -- see PROJECT.md). Something "
                              "at seconds-per-iteration, e.g. '28s/it', is stuck (OOM swapping, GPU not "
                              "actually in use, etc.), not just 'a big model'. Pass 0 or a negative "
                              "number to disable this check entirely.")
    args = parser.parse_args()
    min_it_per_sec = args.min_it_per_sec if args.min_it_per_sec > 0 else None

    all_models = args.models if args.models else list(MODEL_REGISTRY.keys())
    to_run = [m for m in all_models if m not in args.skip]

    log = load_run_log() if args.resume else {}

    if args.resume:
        already_done = [m for m, e in log.items() if e["status"] == "success"]
        to_run = [m for m in to_run if m not in already_done]
        if already_done:
            print(f"Resuming: skipping {len(already_done)} already-successful model(s): {already_done}")

    print(f"Will train {len(to_run)} model(s): {to_run}\n")

    for i, model_name in enumerate(to_run, 1):
        print(f"\n[{i}/{len(to_run)}] {model_name}")
        status, message = train_one(model_name, args.epochs, args.batch_size, args.lr, args.patience,
                                     min_it_per_sec)
        log[model_name] = {"status": status, "message": message}
        save_run_log(log)  # persist after every model, not just at the end

        if status == "failed":
            print(f"  !! {model_name} FAILED: {message} -- continuing with next model")
        else:
            print(f"  OK {model_name}: {message}")

    print(f"\n{'='*70}\nDONE. {sum(1 for e in log.values() if e['status']=='success')}/{len(log)} succeeded.\n{'='*70}")

    table = build_comparison_table(log)
    table_path = OUTPUT_DIR / "comparison_table.csv"
    table.to_csv(table_path, index=False)
    print(f"\nComparison table saved -> {table_path}")
    print(table.to_string(index=False))

    failed = [m for m, e in log.items() if e["status"] == "failed"]
    if failed:
        print(f"\nFailed models (see {RUN_LOG_PATH} for details): {failed}")
        print("Re-run with --resume after fixing (e.g. lower --batch-size) to retry only what's missing.")


if __name__ == "__main__":
    main()
