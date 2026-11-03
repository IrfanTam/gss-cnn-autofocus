"""
rerun_unitfix.py -- Re-run Tahap 3 (offline replay) with a unit-consistent
position axis, plus stronger baselines and paired statistics.

WHY THIS EXISTS (see INNAS notes):
  Dataset_93FOV_101img_1incomplete stores positions as raw servo counts
  (files named sample_pos_443.jpg, step = 3 counts, range ~518-818), while the
  other four sessions store angle in degrees (angle125.00.png, step = 0.5 deg).
  The original simulation applied theta_min = delta = 1.0 to BOTH scales, so the
  same tolerance meant ~0.29 deg in one session and 1.0 deg in the others.
  This script converts the raw counts to degrees (Dynamixel AX-12A:
  300 deg / 1023 counts) BEFORE any simulation, so theta_min and delta mean the
  same physical angle in every session.

  !! ASSUMPTION TO VERIFY BY THE AUTHOR !!  The conversion factor assumes the
  93FOV positions are AX-12A raw position counts (0-1023 over 300 deg).
  Check against the capture code of that session. If the unit is different,
  change RAW_TO_DEG below.

Original files are NOT modified. Outputs go to output_unitfix/.

Usage (run from this folder, in the env that has pyarrow):
    python rerun_unitfix.py              # full run incl. GSS+CNN (needs cnn_cache parquet)
    python rerun_unitfix.py --no-cnn     # CNN-free strategies only
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from gss_simulator import (FOVCurve, SimResult, evaluate_success, simulate_full_sweep,
                           simulate_hill_climbing, simulate_gss_only, simulate_gss_cnn)

HERE = Path(__file__).parent
DATA_PATH = HERE / "output" / "simulation_fov_data.csv"
CACHE_DIR = HERE / "output" / "cnn_cache"
OUT = HERE / "output_unitfix"

RAW_TO_DEG = 300.0 / 1023.0          # AX-12A: 1023 counts = 300 deg  (ASSUMPTION)
RAW_UNIT_DATASETS = ("93FOV",)       # substring match on dataset name
THETA_MIN = 1.0                      # deg
DELTAS = (0.5, 1.0, 2.0)             # deg, success tolerance sensitivity
DELTA_MAIN = 1.0
P_MIN_DEFAULT = 0.86
P_MIN_GRID = (0.50, 0.60, 0.70, 0.80, 0.86, 0.90, 0.95, 0.99)
PER_STEP_SEC = 0.65                  # ESTIMATE, to be replaced by measured latency
TRIGGER_CLASS = "Focused"
BACKBONES = ["swin_tiny", "regnety_400mf", "convnext_tiny", "mobilenetv3_large",
             "resnet18", "efficientnet_b0", "mobilenetv3_small", "efficientnet_b1",
             "densenet121", "resnet34", "shufflenet_v2", "yolo26n", "vit_tiny",
             "focusnet_nano", "focusnet_micro"]


# ---------------------------------------------------------------- baselines
def simulate_hill_climbing_adaptive(curve, start_step=10, delta=1.0):
    """Stronger hill-climbing baseline: start at the lowest position, move
    `start_step` grid points while the score improves; on a decrease reverse
    direction and halve the step; stop when the step reaches 1 and no
    neighbour improves. Evaluations are counted as visited positions."""
    n = len(curve.positions)
    idx, step, direction = 0, start_step, +1
    best = curve.brenner[idx]
    visited = {idx}
    steps = 1
    while True:
        nxt = idx + direction * step
        if nxt < 0 or nxt >= n:
            if step == 1:
                break
            step = max(1, step // 2)
            direction = -direction
            continue
        if nxt not in visited:
            visited.add(nxt)
            steps += 1
        if curve.brenner[nxt] > best:
            idx, best = nxt, curve.brenner[nxt]
        else:
            if step == 1:
                break
            step = max(1, step // 2)
            direction = -direction
    r = SimResult(strategy="hill_climb_adaptive", dataset=curve.dataset, fov_id=curve.fov_id,
                  final_pos=float(curve.positions[idx]), ground_truth_pos=curve.ground_truth_pos,
                  n_steps=steps, trigger_reason="hill_climb_adaptive")
    return evaluate_success(r, delta)


def simulate_coarse_to_fine(curve, n_coarse=10, delta=1.0):
    """Uniform coarse scan (n_coarse points) then exhaustive scan of the
    neighbouring coarse interval. A common non-learning two-stage baseline."""
    n = len(curve.positions)
    coarse_idx = np.unique(np.linspace(0, n - 1, n_coarse).round().astype(int))
    best_c = coarse_idx[np.argmax(curve.brenner[coarse_idx])]
    lo = coarse_idx[max(0, int(np.where(coarse_idx == best_c)[0][0]) - 1)]
    hi = coarse_idx[min(len(coarse_idx) - 1, int(np.where(coarse_idx == best_c)[0][0]) + 1)]
    fine = np.arange(lo, hi + 1)
    idx = fine[np.argmax(curve.brenner[fine])]
    steps = len(coarse_idx) + len(set(fine) - set(coarse_idx))
    r = SimResult(strategy="coarse_to_fine", dataset=curve.dataset, fov_id=curve.fov_id,
                  final_pos=float(curve.positions[idx]), ground_truth_pos=curve.ground_truth_pos,
                  n_steps=steps, trigger_reason="coarse_to_fine")
    return evaluate_success(r, delta)


# ---------------------------------------------------------------- helpers
def load_curves():
    df = pd.read_csv(DATA_PATH)
    df["fov_id"] = df["fov_id"].astype(str)
    raw = df["dataset"].apply(lambda s: any(k in s for k in RAW_UNIT_DATASETS))
    df["pos_deg"] = np.where(raw, df["pos"] * RAW_TO_DEG, df["pos"])
    curves = []
    for (ds, fov), g in df.groupby(["dataset", "fov_id"]):
        g = g.sort_values("pos_deg")
        curves.append(FOVCurve(dataset=ds, fov_id=fov, positions=g["pos_deg"].values,
                               brenner=g["brenner"].values, image_paths=g["abs_filepath"].values))
    return curves


def build_cnn_lookup(model):
    p = CACHE_DIR / f"{model}_predictions.parquet"
    df = pd.read_parquet(p).drop_duplicates(subset="abs_filepath", keep="first")
    lk = df.set_index("abs_filepath")[["pred_class", "confidence"]].to_dict("index")
    return lambda path: ((lk[path]["pred_class"], lk[path]["confidence"]) if path in lk else ("Blur", 0.0))


def row(r, name, delta):
    return dict(strategy=name, dataset=r.dataset, fov_id=r.fov_id, final_pos=r.final_pos,
                ground_truth_pos=r.ground_truth_pos, n_steps=r.n_steps,
                trigger_reason=r.trigger_reason, error_deg=r.error, delta=delta,
                success=bool(r.error <= delta))


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(x, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--p-min", type=float, default=P_MIN_DEFAULT)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    curves = load_curves()
    print(f"{len(curves)} FOV loaded (raw-unit sessions converted: {RAW_UNIT_DATASETS}, x{RAW_TO_DEG:.5f})")

    rows = []
    for delta in DELTAS:
        for c in curves:
            for name, fn in (("full_sweep", lambda c: simulate_full_sweep(c, delta=delta)),
                             ("hill_climbing", lambda c: simulate_hill_climbing(c, delta=delta)),
                             ("hill_climb_adaptive", lambda c: simulate_hill_climbing_adaptive(c, delta=delta)),
                             ("coarse_to_fine", lambda c: simulate_coarse_to_fine(c, delta=delta)),
                             ("gss_only", lambda c: simulate_gss_only(c, theta_min=THETA_MIN, delta=delta))):
                rows.append(row(fn(c), name, delta))
    if not args.no_cnn:
        for m in BACKBONES:
            pred = build_cnn_lookup(m)
            for delta in DELTAS:
                for c in curves:
                    r = simulate_gss_cnn(c, pred, theta_min=THETA_MIN, p_min=args.p_min,
                                         trigger_class=TRIGGER_CLASS, delta=delta)
                    rows.append(row(r, f"gss_cnn_{m}", delta))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "per_fov.csv", index=False)

    g = d.groupby(["delta", "strategy"])
    s = g.agg(n_fov=("success", "size"), success_pct=("success", lambda x: 100 * x.mean()),
              mean_steps=("n_steps", "mean"), std_steps=("n_steps", "std"),
              mean_error_deg=("error_deg", "mean")).reset_index()
    s["est_time_s"] = s["mean_steps"] * PER_STEP_SEC
    s.round(3).to_csv(OUT / "summary.csv", index=False)

    m = d[d.delta == DELTA_MAIN]
    ps = m.groupby(["dataset", "strategy"]).agg(n_fov=("success", "size"),
                                                success_pct=("success", lambda x: 100 * x.mean()),
                                                mean_steps=("n_steps", "mean")).reset_index()
    ps.round(3).to_csv(OUT / "per_session.csv", index=False)

    m = m.assign(key=m.dataset + "|" + m.fov_id)
    S = m.pivot_table(index="key", columns="strategy", values="success", aggfunc="first").astype(float)
    N = m.pivot_table(index="key", columns="strategy", values="n_steps", aggfunc="first")
    base = "gss_only"
    out = []
    for strat in S.columns:
        if strat == base:
            continue
        b = int(((S[base] == 1) & (S[strat] == 0)).sum())
        cc = int(((S[base] == 0) & (S[strat] == 1)).sum())
        dn = (N[base] - N[strat]).values
        lo, hi = boot_ci(dn)
        slo, shi = boot_ci(S[strat].values)
        out.append(dict(strategy=strat, success_pct=100 * S[strat].mean(),
                        success_ci_lo=100 * slo, success_ci_hi=100 * shi,
                        gss_only_only_correct=b, strategy_only_correct=cc,
                        mcnemar_exact_p=mcnemar_exact(b, cc),
                        mean_steps=N[strat].mean(), steps_saved_vs_gss=dn.mean(),
                        steps_saved_ci_lo=lo, steps_saved_ci_hi=hi,
                        steps_saved_pct=100 * dn.mean() / N[base].mean()))
    pd.DataFrame(out).round(3).to_csv(OUT / "paired_stats_vs_gss_only.csv", index=False)

    if not args.no_cnn:
        sweep = []
        for mdl in ("focusnet_nano", "yolo26n", "convnext_tiny", "resnet18"):
            pred = build_cnn_lookup(mdl)
            for pm in P_MIN_GRID:
                rs = [simulate_gss_cnn(c, pred, theta_min=THETA_MIN, p_min=pm,
                                       trigger_class=TRIGGER_CLASS, delta=DELTA_MAIN) for c in curves]
                sweep.append(dict(model=mdl, p_min=pm,
                                  success_pct=100 * np.mean([r.success for r in rs]),
                                  mean_steps=np.mean([r.n_steps for r in rs]),
                                  mean_error_deg=np.mean([r.error for r in rs])))
        pd.DataFrame(sweep).round(3).to_csv(OUT / "sensitivity_p_min_unitfix.csv", index=False)
    print("Done ->", OUT)
    print(s[s.delta == DELTA_MAIN].sort_values("success_pct", ascending=False).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
