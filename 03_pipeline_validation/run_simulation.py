"""
Orchestrates the full autofocus strategy simulation: Full Sweep, Hill
Climbing, GSS-only (all backbone-agnostic, run once) and GSS+CNN dual-
criterion termination (run once per backbone, using cached predictions).

See PROJECT.md ("Tahap 3") and gss_simulator.py for the simulation
semantics and its offline-replay limitations.

Usage:
    python run_simulation.py                      # all 15 backbones
    python run_simulation.py --models resnet18 regnety_400mf
    python run_simulation.py --p-min 0.86          # override default p_min
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gss_simulator import (
    FOVCurve, simulate_full_sweep, simulate_hill_climbing,
    simulate_gss_only, simulate_gss_cnn,
)

DATA_PATH = Path(__file__).parent / "output" / "simulation_fov_data.csv"
CACHE_DIR = Path(__file__).parent / "output" / "cnn_cache"
OUTPUT_DIR = Path(__file__).parent / "output"

# 15 backbones reported in Method Validation -- FocusNet-Small excluded per
# PROJECT.md decision (dropped from reporting, cache kept but unused here).
BACKBONES = [
    "swin_tiny", "regnety_400mf", "convnext_tiny", "mobilenetv3_large",
    "resnet18", "efficientnet_b0", "mobilenetv3_small", "efficientnet_b1",
    "densenet121", "resnet34", "shufflenet_v2", "yolo26n", "vit_tiny",
    "focusnet_nano", "focusnet_micro",
]

THETA_MIN = 1.0
DELTA = 1.0
P_MIN_DEFAULT = 0.86
PER_STEP_SEC = 0.65
TRIGGER_CLASS = "Focused"


def load_curves():
    df = pd.read_csv(DATA_PATH)
    df["fov_id"] = df["fov_id"].astype(str)
    curves = []
    for (dataset, fov_id), g in df.groupby(["dataset", "fov_id"]):
        g = g.sort_values("pos")
        curves.append(FOVCurve(
            dataset=dataset, fov_id=fov_id,
            positions=g["pos"].values, brenner=g["brenner"].values,
            image_paths=g["abs_filepath"].values,
        ))
    return curves


def build_cnn_lookup(model_name):
    """Returns a callable(image_path) -> (pred_class, confidence) backed by
    the precomputed cache, so simulate_gss_cnn() never re-runs inference."""
    cache_path = CACHE_DIR / f"{model_name}_predictions.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(f"No cache for {model_name}: {cache_path}")
    df = pd.read_parquet(cache_path)
    # abs_filepath can repeat (e.g. identical scanned position appearing in
    # more than one FOV group boundary) -- keep the first occurrence, they
    # are predictions for the same physical image so any duplicate is
    # redundant, not conflicting.
    df = df.drop_duplicates(subset="abs_filepath", keep="first")
    lookup = df.set_index("abs_filepath")[["pred_class", "confidence"]].to_dict("index")

    def predict(image_path):
        row = lookup.get(image_path)
        if row is None:
            return "Blur", 0.0  # conservative fallback, should not happen if cache is complete
        return row["pred_class"], row["confidence"]

    return predict


def summarize(results, strategy_name):
    n = len(results)
    success_rate = np.mean([r.success for r in results]) * 100
    mean_steps = np.mean([r.n_steps for r in results])
    std_steps = np.std([r.n_steps for r in results])
    mean_time = mean_steps * PER_STEP_SEC
    mean_error = np.mean([r.error for r in results])

    trigger_counts = pd.Series([r.trigger_reason for r in results]).value_counts()
    trigger_pct = (trigger_counts / n * 100).round(1).to_dict()

    return {
        "strategy": strategy_name,
        "n_fov": n,
        "success_rate_pct": round(success_rate, 1),
        "mean_steps": round(mean_steps, 2),
        "std_steps": round(std_steps, 2),
        "mean_estimated_time_sec": round(mean_time, 2),
        "mean_error_deg": round(mean_error, 3),
        "trigger_breakdown_pct": trigger_pct,
    }


def run_backbone_agnostic_strategies(curves):
    """Full Sweep, Hill Climbing, GSS-only don't depend on any CNN -- run once."""
    results = {"full_sweep": [], "hill_climbing": [], "gss_only": []}
    per_fov_rows = []

    for curve in curves:
        fs = simulate_full_sweep(curve, delta=DELTA)
        hc = simulate_hill_climbing(curve, delta=DELTA)
        go = simulate_gss_only(curve, theta_min=THETA_MIN, delta=DELTA)

        results["full_sweep"].append(fs)
        results["hill_climbing"].append(hc)
        results["gss_only"].append(go)

        for r in (fs, hc, go):
            per_fov_rows.append(dict(
                strategy=r.strategy, dataset=r.dataset, fov_id=r.fov_id,
                final_pos=r.final_pos, ground_truth_pos=r.ground_truth_pos,
                n_steps=r.n_steps, trigger_reason=r.trigger_reason,
                success=r.success, error=r.error,
            ))

    return results, per_fov_rows


def run_gss_cnn_for_backbone(curves, model_name, p_min):
    cnn_predict = build_cnn_lookup(model_name)
    results = []
    per_fov_rows = []

    for curve in curves:
        r = simulate_gss_cnn(curve, cnn_predict, theta_min=THETA_MIN, p_min=p_min,
                              trigger_class=TRIGGER_CLASS, delta=DELTA)
        results.append(r)
        per_fov_rows.append(dict(
            strategy=f"gss_cnn_{model_name}", dataset=r.dataset, fov_id=r.fov_id,
            final_pos=r.final_pos, ground_truth_pos=r.ground_truth_pos,
            n_steps=r.n_steps, trigger_reason=r.trigger_reason,
            success=r.success, error=r.error,
        ))

    return results, per_fov_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None,
                         help="Subset of backbones to simulate (default: all 15)")
    parser.add_argument("--p-min", type=float, default=P_MIN_DEFAULT)
    args = parser.parse_args()

    backbones = args.models if args.models else BACKBONES

    print("Loading FOV curves...")
    curves = load_curves()
    print(f"Loaded {len(curves)} FOV.")

    print("\n=== Backbone-agnostic strategies (Full Sweep, Hill Climbing, GSS-only) ===")
    agnostic_results, agnostic_rows = run_backbone_agnostic_strategies(curves)

    summary_rows = []
    for strategy, results in agnostic_results.items():
        s = summarize(results, strategy)
        summary_rows.append(s)
        print(f"  {strategy:15s}: success={s['success_rate_pct']:5.1f}%  "
              f"steps={s['mean_steps']:6.2f}±{s['std_steps']:5.2f}  "
              f"time={s['mean_estimated_time_sec']:6.2f}s  error={s['mean_error_deg']:.3f}")

    print(f"\n=== GSS + CNN dual-criterion (p_min={args.p_min}) ===")
    all_rows = list(agnostic_rows)

    for model_name in backbones:
        try:
            results, rows = run_gss_cnn_for_backbone(curves, model_name, args.p_min)
        except FileNotFoundError as e:
            print(f"  !! {model_name}: SKIPPED ({e})")
            continue

        s = summarize(results, f"gss_cnn_{model_name}")
        summary_rows.append(s)
        all_rows.extend(rows)
        print(f"  {model_name:20s}: success={s['success_rate_pct']:5.1f}%  "
              f"steps={s['mean_steps']:6.2f}±{s['std_steps']:5.2f}  "
              f"time={s['mean_estimated_time_sec']:6.2f}s  error={s['mean_error_deg']:.3f}  "
              f"trigger={s['trigger_breakdown_pct']}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "simulation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved -> {summary_path}")

    detail_df = pd.DataFrame(all_rows)
    detail_path = OUTPUT_DIR / "simulation_per_fov.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"Per-FOV detail saved -> {detail_path}")


if __name__ == "__main__":
    main()
