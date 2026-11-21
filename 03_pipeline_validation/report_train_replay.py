"""
Supplementary analysis (Step T3): replay GSS-only and GSS+CNN on the 299 TRAIN FOV,
using the thresholds already selected on the validation split
(output_selection/val_selection.csv). Nothing is re-selected here.

Output: output_train/train_report.csv with success, mean steps, saved steps and
mean error per classifier. Train FOV were seen by the classifiers, so these numbers
are an optimistic reference, not a result to be used for model selection.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from gss_simulator import simulate_gss_only  # noqa: E402
from rerun_unitfix import THETA_MIN, DELTA_MAIN, PER_STEP_SEC, mcnemar_exact, boot_ci  # noqa: E402
from select_on_val import load_curves, lookup, run  # noqa: E402


def main():
    curves = load_curves(HERE / "output_train" / "simulation_fov_data_train.csv")
    sel = pd.read_csv(HERE / "output_selection" / "val_selection.csv")
    print(f"train FOV: {len(curves)}")
    gss = [simulate_gss_only(c, theta_min=THETA_MIN, delta=DELTA_MAIN) for c in curves]
    g_succ = np.array([r.success for r in gss], float)
    g_steps = np.array([r.n_steps for r in gss], float)
    print(f"GSS-only: {100*g_succ.mean():.1f}% success, {g_steps.mean():.2f} steps")
    rows = []
    for _, s in sel.iterrows():
        rs = run(curves, lookup(HERE / "output_train" / "cnn_cache", s.model), s.p_min)
        succ = np.array([r.success for r in rs], float)
        steps = np.array([r.n_steps for r in rs], float)
        d = g_steps - steps
        lo, hi = boot_ci(d)
        b = int(((g_succ == 1) & (succ == 0)).sum()); cc = int(((g_succ == 0) & (succ == 1)).sum())
        rows.append(dict(model=s.model, p_min_val_selected=s.p_min, n_fov=len(curves),
                         train_success_pct=100 * succ.mean(), train_mean_steps=steps.mean(),
                         est_time_s=steps.mean() * PER_STEP_SEC,
                         steps_saved_pct=100 * d.mean() / g_steps.mean(),
                         steps_saved_ci_lo=lo, steps_saved_ci_hi=hi,
                         gss_only_only_correct=b, model_only_correct=cc,
                         mcnemar_exact_p=mcnemar_exact(b, cc),
                         mean_error_deg=np.mean([r.error for r in rs])))
    rep = pd.DataFrame(rows).sort_values("train_mean_steps")
    rep.round(3).to_csv(HERE / "output_train" / "train_report.csv", index=False)
    print(rep.round(2).to_string(index=False))
    print("Saved -> output_train/train_report.csv")


if __name__ == "__main__":
    main()
