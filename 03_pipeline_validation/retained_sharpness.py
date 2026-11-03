"""
retained_sharpness.py -- How sharp is the frame where the search stops?
For every test FOV and every classifier (at its VALIDATION-selected p_min from
output_selection/val_selection.csv), record the final position and the
normalised Brenner score there: P_final = B(final)/B_max  (1.0 = at the peak).
Also reports the share of stops with P_final > 0.55 (the 'Focused' label limit)
and > 0.30. No new selection is made; test data are only reported.

Run (same env as select_on_val.py, needs pyarrow):
    python retained_sharpness.py
Outputs -> output_selection/retained_sharpness_per_fov.csv, retained_sharpness_summary.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from select_on_val import load_curves, lookup, run  # noqa: E402
from gss_simulator import simulate_gss_only  # noqa: E402
from rerun_unitfix import THETA_MIN, DELTA_MAIN  # noqa: E402

out = HERE / "output_selection"
curves = load_curves(str(HERE / "output" / "simulation_fov_data.csv"))
sel = pd.read_csv(out / "val_selection.csv")
cidx = {(c.dataset, c.fov_id): c for c in curves}

def p_at(c, pos):
    i = int(np.argmin(np.abs(c.positions - pos)))
    return float(c.brenner[i] / c.brenner.max())

rows = []
for r in [simulate_gss_only(c, theta_min=THETA_MIN, delta=DELTA_MAIN) for c in curves]:
    rows.append(dict(model="gss_only", dataset=r.dataset, fov_id=r.fov_id, n_steps=r.n_steps,
                     error_deg=r.error, success=bool(r.success), final_pos=float(r.final_pos), z_star=float(r.ground_truth_pos), p_final=p_at(cidx[(r.dataset, r.fov_id)], r.final_pos)))
for _, s in sel.iterrows():
    pred = lookup(HERE / "output" / "cnn_cache", s.model)
    for r in run(curves, pred, s.p_min):
        rows.append(dict(model=s.model, dataset=r.dataset, fov_id=r.fov_id, n_steps=r.n_steps,
                         error_deg=r.error, success=bool(r.success), final_pos=float(r.final_pos), z_star=float(r.ground_truth_pos), p_final=p_at(cidx[(r.dataset, r.fov_id)], r.final_pos)))
d = pd.DataFrame(rows)
d.to_csv(out / "retained_sharpness_per_fov.csv", index=False)
g = d.groupby("model")
summ = pd.DataFrame({"mean_p_final": g.p_final.mean(), "min_p_final": g.p_final.min(),
                     "pct_above_0.55": 100 * g.p_final.apply(lambda x: (x > 0.55).mean()),
                     "pct_above_0.30": 100 * g.p_final.apply(lambda x: (x > 0.30).mean()),
                     "n_success": g.success.sum()}).round(3)
summ.to_csv(out / "retained_sharpness_summary.csv")
print(summ.to_string())
f = d[(d.model != "gss_only") & (~d.success)]
print("\nFailed runs (outside delta): n =", len(f), " mean P_final =", round(f.p_final.mean(), 3),
      " share P_final>0.55 =", round(100 * (f.p_final > 0.55).mean(), 1), "%")
