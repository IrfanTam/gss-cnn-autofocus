"""R1-7: regenerate the archive tables (splits, selection, per-run log, baselines) from fixed inputs."""
import json, os, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
EXP = Path(__file__).resolve().parent.parent; P3 = EXP / "03_pipeline_validation"; sys.path.insert(0, str(P3)); sys.path.insert(0, str(EXP / "01_data_preparation"))
from gss_simulator import PHI, FOVCurve
from select_on_val import load_curves
from rerun_unitfix import build_cnn_lookup, THETA_MIN
OUT = Path(__file__).parent; PREP = EXP / "01_data_preparation" / "output"
NAMES = ['efficientnet_b1','resnet18','mobilenetv3_small','resnet34','vit_tiny','yolo26n','densenet121','regnety_400mf','mobilenetv3_large','efficientnet_b0','convnext_tiny','swin_tiny','shufflenet_v2','focusnet_nano','focusnet_micro']
# ---- splits (anonymised field codes; slide identity was not recorded)
rows = []
for s in ("train", "val", "test"):
    d = pd.read_csv(PREP / f"{s}_balanced.csv")
    for (ds, f), g in d.groupby(["dataset", d.fov_id.astype(str)]): rows.append(dict(dataset=ds, fov_id=f, split=s))
sp = pd.DataFrame(rows)
XJ = EXP.parent.parent / "Analisa Dataset" / "excluded_fov.json"
XJ = XJ if XJ.exists() else Path(os.path.expanduser("~/mnt/Bu Heny/Analisa Dataset/excluded_fov.json"))
excl = json.load(open(XJ))
ex = pd.DataFrame([dict(dataset=k, fov_id=str(v), split="excluded") for k, vs in excl.items() for v in vs])
sp = pd.concat([sp, ex], ignore_index=True)
rng = np.random.default_rng(20261120); order = rng.permutation(len(sp)); sp["field_code"] = ["F%03d" % (i + 1) for i in np.argsort(order)]
sp["session"] = sp.dataset; sp["slide_group"] = "not recorded"
sp.sort_values("field_code")[["field_code", "session", "slide_group", "split"]].to_csv(OUT / "splits_public.csv", index=False)
sp.to_csv(OUT / "field_key_map_PRIVATE.csv", index=False)
code = {(r.dataset, r.fov_id): r.field_code for r in sp.itertuples()}
# ---- selection
rep = pd.read_csv(P3 / "output_selection/test_report.csv"); vs = pd.read_csv(P3 / "output_selection/val_selection.csv") if (P3 / "output_selection/val_selection.csv").exists() else None
ck = lambda m: f"02_backbone_training/output/yolo26n_balanced/weights/best.pt" if m == "yolo26n" else f"02_backbone_training/output/{m}_balanced/{m}_best.pt"
sel = rep[["model", "p_min_val_selected", "meets_val_constraint", "selected_backbone", "test_success_pct", "test_mean_steps"]].copy(); sel["checkpoint"] = sel.model.map(ck); sel.to_csv(OUT / "classifier_selection.csv", index=False)
# ---- per-run log
curves = [c for c in load_curves(P3 / "output/simulation_fov_data.csv")]
pm = dict(zip(rep.model, rep.p_min_val_selected)); log = []
def run(c, fn, name, pmin):
    a, b = c.pos_min, c.pos_max; req = []; samp = []
    def cap(x): req.append(round(float(x), 3)); samp.append(float(c.positions[c.nearest_index(x)]))
    cc = b - (b - a) / PHI; dd = a + (b - a) / PHI; cap(cc); cap(dd); fc, fd = c.brenner_at(cc), c.brenner_at(dd)
    stop = None
    trig = lambda x: fn is not None and (lambda r: r[0] == "Focused" and r[1] >= pmin)(fn(c.image_at(x)))
    if fn is not None and (trig(cc) or trig(dd)): stop = "classifier"; fin = cc if trig(cc) else dd
    while stop is None and (b - a) > THETA_MIN:
        if fc > fd: b = dd; dd, fd = cc, fc; cc = b - (b - a) / PHI; fc = c.brenner_at(cc); new = cc
        else: a = cc; cc, fc = dd, fd; dd = a + (b - a) / PHI; fd = c.brenner_at(dd); new = dd
        cap(new)
        if trig(new): stop = "classifier"; fin = new
    if stop is None: stop = "interval"; fin = (a + b) / 2; cap(fin)
    fs = float(c.positions[c.nearest_index(fin)])
    return dict(strategy=name, field_code=code[(c.dataset, c.fov_id)], requested_angles=";".join(map(str, req)), sampled_angles=";".join(map(str, samp)), stop_reason=stop, final_requested=round(float(fin), 3), final_sampled=fs, charged_steps=len(req), distinct_recorded_frames=len(set(samp)), final_error_deg=round(abs(fs - c.ground_truth_pos), 3), success_1deg=abs(fs - c.ground_truth_pos) <= 1.0, time_estimate_s=round(len(req) * 0.65, 2))
for c in curves:
    log.append(run(c, None, "gss_only", None))
for m in NAMES:
    fn = build_cnn_lookup(m)
    for c in curves: log.append(run(c, fn, "gss_" + m, pm[m]))
L = pd.DataFrame(log); L.to_csv(OUT / "per_run_test.csv", index=False)
# ---- consistency with the official outputs
t = pd.read_csv(P3 / "output_selection/test_per_fov.csv"); t["field_code"] = [code[(a, str(b))] for a, b in zip(t.dataset, t.fov_id)]
chk = L[L.strategy != "gss_only"].assign(model=lambda d: d.strategy.str[4:]).merge(t, on=["model", "field_code"])
assert (chk.charged_steps == chk.n_steps).all() and (chk.success_1deg == chk.success).all(), "per-run log differs from the official replay"
print("per-run log consistent with the official replay:", len(chk), "runs; fields:", len(curves))
# ---- baselines
u = pd.read_csv(P3 / "output_unitfix/per_fov.csv"); u = u[(u.delta == 1.0) & u.strategy.isin(["full_sweep", "hill_climbing", "hill_climb_adaptive", "coarse_to_fine", "gss_only"])].copy()
u["field_code"] = [code[(a, str(b))] for a, b in zip(u.dataset, u.fov_id)]
u[["strategy", "field_code", "final_pos", "ground_truth_pos", "n_steps", "trigger_reason", "error_deg", "success"]].to_csv(OUT / "baselines_per_field.csv", index=False)
print("archive written to", OUT)
