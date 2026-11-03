"""
Step C. Select p_min (per backbone) and the final backbone on the VALIDATION
FOV, then report the TEST FOV exactly once.

Protocol (fixed before looking at the test numbers):
  1. Positions of raw-count sessions are converted to degrees (see rerun_unitfix.py;
     the 300/1023 factor is an ASSUMPTION that must be verified by the author).
  2. For every backbone and every p_min in P_MIN_GRID, simulate GSS+CNN on the val FOV.
  3. A (backbone, p_min) pair is ACCEPTABLE if its val success count is at least
     (GSS-only val success count - SLACK_FOV).   [SLACK_FOV = 1 FOV]
  4. Per backbone: pick the acceptable p_min with the fewest mean steps (ties: lower
     mean error). If none is acceptable, pick the p_min with the highest success
     (flagged meets_val_constraint = False).
  5. Final backbone: among backbones that meet the constraint, the fewest val steps.
  6. Report test FOV (71) for ALL backbones at their val-chosen p_min, paired against
     GSS-only (exact McNemar, bootstrap CIs).

Outputs -> output_selection/: val_grid.csv, val_selection.csv, test_report.csv,
selected.json, test_per_fov.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from gss_simulator import FOVCurve, simulate_gss_only, simulate_gss_cnn  # noqa: E402
from rerun_unitfix import (RAW_TO_DEG, RAW_UNIT_DATASETS, THETA_MIN, DELTA_MAIN, P_MIN_GRID,  # noqa: E402
                           BACKBONES, TRIGGER_CLASS, PER_STEP_SEC, mcnemar_exact, boot_ci)

SLACK_FOV = 1


def load_curves(path):
    df = pd.read_csv(path)
    df["fov_id"] = df["fov_id"].astype(str)
    raw = df["dataset"].apply(lambda s: any(k in s for k in RAW_UNIT_DATASETS))
    df["pos_deg"] = np.where(raw, df["pos"] * RAW_TO_DEG, df["pos"])
    curves = []
    for (ds, fov), g in df.groupby(["dataset", "fov_id"]):
        g = g.sort_values("pos_deg")
        curves.append(FOVCurve(dataset=ds, fov_id=fov, positions=g["pos_deg"].values,
                               brenner=g["brenner"].values, image_paths=g["abs_filepath"].values))
    return curves


def lookup(cache_dir, model):
    df = pd.read_parquet(Path(cache_dir) / f"{model}_predictions.parquet")
    df = df.drop_duplicates(subset="abs_filepath", keep="first")
    lk = df.set_index("abs_filepath")[["pred_class", "confidence"]].to_dict("index")
    return lambda p: ((lk[p]["pred_class"], lk[p]["confidence"]) if p in lk else ("Blur", 0.0))


def run(curves, pred, p_min):
    return [simulate_gss_cnn(c, pred, theta_min=THETA_MIN, p_min=p_min,
                             trigger_class=TRIGGER_CLASS, delta=DELTA_MAIN) for c in curves]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-data", default=str(HERE / "output_val" / "simulation_fov_data_val.csv"))
    ap.add_argument("--val-cache", default=str(HERE / "output_val" / "cnn_cache"))
    ap.add_argument("--test-data", default=str(HERE / "output" / "simulation_fov_data.csv"))
    ap.add_argument("--test-cache", default=str(HERE / "output" / "cnn_cache"))
    ap.add_argument("--out", default=str(HERE / "output_selection"))
    ap.add_argument("--models", nargs="*", default=BACKBONES)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    val_curves, test_curves = load_curves(a.val_data), load_curves(a.test_data)
    vk = {(c.dataset, c.fov_id) for c in val_curves}
    tk = {(c.dataset, c.fov_id) for c in test_curves}
    assert not (vk & tk), "val and test FOV overlap -- selection would leak into the test report"
    print(f"val FOV: {len(val_curves)}, test FOV: {len(test_curves)}, disjoint")

    # ---- validation: GSS-only reference and full (backbone x p_min) grid
    gss_val = [simulate_gss_only(c, theta_min=THETA_MIN, delta=DELTA_MAIN) for c in val_curves]
    gss_val_ok = int(sum(r.success for r in gss_val))
    print(f"GSS-only on val: {gss_val_ok}/{len(val_curves)} success, "
          f"{np.mean([r.n_steps for r in gss_val]):.2f} steps")
    grid = []
    for m in a.models:
        pred = lookup(a.val_cache, m)
        for pm in P_MIN_GRID:
            rs = run(val_curves, pred, pm)
            grid.append(dict(model=m, p_min=pm, n_success=int(sum(r.success for r in rs)),
                             success_pct=100 * np.mean([r.success for r in rs]),
                             mean_steps=np.mean([r.n_steps for r in rs]),
                             mean_error_deg=np.mean([r.error for r in rs])))
    grid = pd.DataFrame(grid)
    grid.round(3).to_csv(out / "val_grid.csv", index=False)

    # ---- per-backbone selection on val
    sel = []
    for m, g in grid.groupby("model", sort=False):
        ok = g[g.n_success >= gss_val_ok - SLACK_FOV]
        if len(ok):
            pick, flag = ok.sort_values(["mean_steps", "mean_error_deg"]).iloc[0], True
        else:
            pick, flag = g.sort_values(["n_success", "mean_steps"], ascending=[False, True]).iloc[0], False
        sel.append(dict(model=m, p_min=pick.p_min, meets_val_constraint=flag,
                        val_success_pct=pick.success_pct, val_mean_steps=pick.mean_steps,
                        val_mean_error_deg=pick.mean_error_deg))
    sel = pd.DataFrame(sel)
    cand = sel[sel.meets_val_constraint]
    final = (cand.sort_values("val_mean_steps").iloc[0] if len(cand)
             else sel.sort_values(["val_success_pct", "val_mean_steps"], ascending=[False, True]).iloc[0])
    sel["selected_backbone"] = sel.model == final.model
    sel.round(3).to_csv(out / "val_selection.csv", index=False)
    print("\nVal selection (per backbone):")
    print(sel.round(3).to_string(index=False))
    print(f"\n=> Selected backbone on val: {final.model} (p_min={final.p_min})")

    # ---- test report: ONE pass, each backbone at its val-chosen p_min
    gss_test = [simulate_gss_only(c, theta_min=THETA_MIN, delta=DELTA_MAIN) for c in test_curves]
    g_succ = np.array([r.success for r in gss_test], float)
    g_steps = np.array([r.n_steps for r in gss_test], float)
    rows, per_fov = [], []
    for _, s in sel.iterrows():
        rs = run(test_curves, lookup(a.test_cache, s.model), s.p_min)
        succ = np.array([r.success for r in rs], float)
        steps = np.array([r.n_steps for r in rs], float)
        b, cc = int(((g_succ == 1) & (succ == 0)).sum()), int(((g_succ == 0) & (succ == 1)).sum())
        d = g_steps - steps
        lo, hi = boot_ci(d)
        slo, shi = boot_ci(succ)
        rows.append(dict(model=s.model, p_min_val_selected=s.p_min,
                         meets_val_constraint=s.meets_val_constraint,
                         selected_backbone=bool(s.selected_backbone),
                         test_success_pct=100 * succ.mean(), success_ci_lo=100 * slo, success_ci_hi=100 * shi,
                         test_mean_steps=steps.mean(), est_time_s=steps.mean() * PER_STEP_SEC,
                         steps_saved_pct=100 * d.mean() / g_steps.mean(),
                         steps_saved_ci_lo=lo, steps_saved_ci_hi=hi,
                         gss_only_only_correct=b, model_only_correct=cc,
                         mcnemar_exact_p=mcnemar_exact(b, cc),
                         mean_error_deg=np.mean([r.error for r in rs])))
        for r in rs:
            per_fov.append(dict(model=s.model, dataset=r.dataset, fov_id=r.fov_id, n_steps=r.n_steps,
                                trigger=r.trigger_reason, success=r.success, error_deg=r.error))
    rep = pd.DataFrame(rows).sort_values("test_mean_steps")
    rep.round(3).to_csv(out / "test_report.csv", index=False)
    pd.DataFrame(per_fov).to_csv(out / "test_per_fov.csv", index=False)
    json.dump(dict(selected_backbone=final.model, p_min=float(final.p_min), slack_fov=SLACK_FOV,
                   gss_only_val_success=gss_val_ok, n_val_fov=len(val_curves), n_test_fov=len(test_curves),
                   raw_to_deg=RAW_TO_DEG, unit_assumption_verified=False),
              open(out / "selected.json", "w"), indent=2)
    print(f"\nTest (GSS-only: {100 * g_succ.mean():.1f}% success, {g_steps.mean():.2f} steps):")
    print(rep.round(2).to_string(index=False))
    print("\nSaved ->", out)


if __name__ == "__main__":
    main()
