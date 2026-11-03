"""
Sweep p_min (CNN confidence threshold for dual-criterion early-stop) across
a range of values, for representative backbones, to show the success-rate
vs mean-steps trade-off. See PROJECT.md Tahap 3 checklist item 5.

Usage:
    python sensitivity_p_min.py                                  # default backbones
    python sensitivity_p_min.py --models resnet18 focusnet_nano
    python sensitivity_p_min.py --p-min-values 0.5 0.6 0.7 0.8 0.9 0.95 0.99
"""

import argparse
from pathlib import Path

import pandas as pd

from run_simulation import load_curves, run_gss_cnn_for_backbone, summarize

OUTPUT_DIR = Path(__file__).parent / "output"

DEFAULT_MODELS = ["focusnet_nano", "yolo26n", "convnext_tiny", "resnet18"]
DEFAULT_P_MIN_VALUES = [0.50, 0.60, 0.70, 0.80, 0.86, 0.90, 0.95, 0.99]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--p-min-values", nargs="*", type=float, default=DEFAULT_P_MIN_VALUES)
    args = parser.parse_args()

    print("Loading FOV curves...")
    curves = load_curves()
    print(f"Loaded {len(curves)} FOV.\n")

    rows = []
    for model_name in args.models:
        print(f"=== {model_name} ===")
        for p_min in args.p_min_values:
            results, _ = run_gss_cnn_for_backbone(curves, model_name, p_min)
            s = summarize(results, f"gss_cnn_{model_name}")
            s["model"] = model_name
            s["p_min"] = p_min
            rows.append(s)
            print(f"  p_min={p_min:.2f}: success={s['success_rate_pct']:5.1f}%  "
                  f"steps={s['mean_steps']:6.2f}±{s['std_steps']:5.2f}  "
                  f"error={s['mean_error_deg']:.3f}  trigger={s['trigger_breakdown_pct']}")
        print()

    df = pd.DataFrame(rows)
    cols = ["model", "p_min", "success_rate_pct", "mean_steps", "std_steps",
            "mean_estimated_time_sec", "mean_error_deg", "trigger_breakdown_pct"]
    df = df[cols]

    out_path = OUTPUT_DIR / "sensitivity_p_min.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
