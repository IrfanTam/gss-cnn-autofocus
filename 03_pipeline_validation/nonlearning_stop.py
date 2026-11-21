"""
R2-3: non-learning early stop for GSS, using ONLY frames observed so far (no oracle B_max).
Two deployable rules, each with one parameter eps tuned on the VALIDATION FOV with exactly the
protocol of select_on_val.py (acceptable if val success >= GSS-only val success - 1 FOV; fewest
mean steps; if none acceptable, highest val success). The test FOV are evaluated once.

  plateau : stop when the two interior points are equally sharp, |f_c - f_d| / max(f_c, f_d) <= eps
            (a flat-topped bracket); returned position = the sharper of c and d.
  gain    : stop when the best observed score improved by <= eps (relative) over each of the last
            two newly acquired frames; returned position = the best observed frame.
Both fall back to the angular criterion (interval <= theta_min, midpoint returned) as in GSS-only.
Steps are counted as in gss_simulator.py (2 initial + 1 per new frame).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
from gss_simulator import PHI, SimResult, evaluate_success, simulate_gss_only
from select_on_val import load_curves, SLACK_FOV
from rerun_unitfix import THETA_MIN, DELTA_MAIN, PER_STEP_SEC, mcnemar_exact, boot_ci

EPS_GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

def sim(curve, rule, eps, theta_min=THETA_MIN, delta=DELTA_MAIN):
    a, b = curve.pos_min, curve.pos_max
    c = b - (b - a) / PHI; d = a + (b - a) / PHI
    fc, fd = curve.brenner_at(c), curve.brenner_at(d); steps = 2
    best_hist = [max(fc, fd)]; obs = [(c, fc), (d, fd)]
    def stop_now(last_gain_ok):
        if rule == "plateau": return abs(fc - fd) / max(fc, fd, 1e-12) <= eps
        return last_gain_ok
    gains = []
    if rule == "plateau" and stop_now(False):
        pos = c if fc >= fd else d
        return evaluate_success(SimResult(f"nl_{rule}", curve.dataset, curve.fov_id, float(curve.positions[curve.nearest_index(pos)]), curve.ground_truth_pos, steps, "sharpness"), delta)
    while (b - a) > theta_min:
        if fc > fd: b = d; d, fd = c, fc; c = b - (b - a) / PHI; fc = curve.brenner_at(c); new = (c, fc)
        else: a = c; c, fc = d, fd; d = a + (b - a) / PHI; fd = curve.brenner_at(d); new = (d, fd)
        steps += 1; obs.append(new)
        prev_best = best_hist[-1]; best = max(prev_best, new[1]); best_hist.append(best)
        gains.append((best - prev_best) / max(prev_best, 1e-12))
        ok = len(gains) >= 2 and gains[-1] <= eps and gains[-2] <= eps
        if stop_now(ok):
            pos = max(obs, key=lambda t: t[1])[0] if rule == "gain" else (c if fc >= fd else d)
            return evaluate_success(SimResult(f"nl_{rule}", curve.dataset, curve.fov_id, float(curve.positions[curve.nearest_index(pos)]), curve.ground_truth_pos, steps, "sharpness"), delta)
    steps += 1  # returned midpoint is one more captured position (R1-1)
    fp = curve.positions[curve.nearest_index((a + b) / 2)]
    return evaluate_success(SimResult(f"nl_{rule}", curve.dataset, curve.fov_id, float(fp), curve.ground_truth_pos, steps, "angular"), delta)

def main():
    out = HERE / "output_nonlearning"; out.mkdir(exist_ok=True)
    val = load_curves(HERE / "output_val" / "simulation_fov_data_val.csv")
    test = load_curves(HERE / "output" / "simulation_fov_data.csv")
    assert not ({(c.dataset, c.fov_id) for c in val} & {(c.dataset, c.fov_id) for c in test})
    gv = [simulate_gss_only(c, THETA_MIN, DELTA_MAIN) for c in val]; gv_ok = int(sum(r.success for r in gv))
    gt = [simulate_gss_only(c, THETA_MIN, DELTA_MAIN) for c in test]
    g_succ = np.array([r.success for r in gt], float); g_steps = np.array([r.n_steps for r in gt], float)
    print(f"val {len(val)} FOV, GSS-only val success {gv_ok}; test {len(test)} FOV, GSS-only {int(g_succ.sum())}/71, {g_steps.mean():.2f} steps")
    grid, rep, per = [], [], []
    for rule in ("plateau", "gain"):
        rows = []
        for eps in EPS_GRID:
            rs = [sim(c, rule, eps) for c in val]
            rows.append(dict(rule=rule, eps=eps, n_success=int(sum(r.success for r in rs)), mean_steps=np.mean([r.n_steps for r in rs]), mean_error_deg=np.mean([r.error for r in rs])))
        g = pd.DataFrame(rows); grid.append(g)
        ok = g[g.n_success >= gv_ok - SLACK_FOV]
        if len(ok): pick, flag = ok.sort_values(["mean_steps", "mean_error_deg"]).iloc[0], True
        else: pick, flag = g.sort_values(["n_success", "mean_steps"], ascending=[False, True]).iloc[0], False
        rs = [sim(c, rule, pick.eps) for c in test]
        succ = np.array([r.success for r in rs], float); steps = np.array([r.n_steps for r in rs], float)
        b_, c_ = int(((g_succ == 1) & (succ == 0)).sum()), int(((g_succ == 0) & (succ == 1)).sum())
        d_ = g_steps - steps; lo, hi = boot_ci(d_)
        rep.append(dict(rule=rule, eps_val_selected=pick.eps, meets_val_constraint=flag, val_success=int(pick.n_success), val_mean_steps=round(pick.mean_steps, 2),
                        test_success=int(succ.sum()), test_mean_steps=round(steps.mean(), 2), est_time_s=round(steps.mean() * PER_STEP_SEC, 2),
                        steps_saved_pct=round(100 * d_.mean() / g_steps.mean(), 1), gss_only_only_correct=b_, rule_only_correct=c_,
                        mcnemar_exact_p=round(mcnemar_exact(b_, c_), 3), mean_error_deg=round(np.mean([r.error for r in rs]), 2),
                        stop_on_sharpness=int(sum(r.trigger_reason == "sharpness" for r in rs))))
        for r in rs: per.append(dict(rule=rule, dataset=r.dataset, fov_id=r.fov_id, n_steps=r.n_steps, trigger=r.trigger_reason, success=r.success, error_deg=r.error))
    pd.concat(grid).round(3).to_csv(out / "val_grid_nonlearning.csv", index=False)
    pd.DataFrame(per).to_csv(out / "test_per_fov_nonlearning.csv", index=False)
    R = pd.DataFrame(rep); R.to_csv(out / "test_report_nonlearning.csv", index=False)
    print(pd.concat(grid).round(2).to_string(index=False)); print(); print(R.to_string(index=False))

if __name__ == "__main__": main()
