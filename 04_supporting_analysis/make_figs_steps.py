import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.expanduser("~/mnt/Bu Heny/")
B = H + "Method X/experiment/03_pipeline_validation/"; OUT = H + "INNAS/figures/"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 7, "axes.linewidth": 0.6, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "mathtext.fontset": "dejavuserif"})
NAME = {'efficientnet_b1':'EfficientNet-B1','resnet18':'ResNet-18','mobilenetv3_small':'MobileNetV3-S','resnet34':'ResNet-34','vit_tiny':'ViT-Tiny','yolo26n':'YOLO26n-cls','densenet121':'DenseNet-121','regnety_400mf':'RegNetY-400MF','mobilenetv3_large':'MobileNetV3-L','efficientnet_b0':'EfficientNet-B0','convnext_tiny':'ConvNeXt-Tiny','swin_tiny':'Swin-Tiny','shufflenet_v2':'ShuffleNetV2','focusnet_nano':'FocusNet-Nano','focusnet_micro':'FocusNet-Micro'}
rep = pd.read_csv(B + "output_selection/test_report.csv"); sw = pd.read_csv(B + "output_unitfix/sensitivity_p_min_unitfix.csv")
tp = pd.read_csv(B + "output_selection/test_per_fov.csv"); vg = pd.read_csv(B + "output_selection/val_grid.csv")
u = pd.read_csv(B + "output_unitfix/per_fov.csv"); g = u[(u.strategy == "gss_only") & (u.delta == 1.0)]
for f in ("fig_pareto", "fig_steps", "fig_val"):
    p = OUT + f + ".png"; q = OUT + f + "_v5steps.png"
    if os.path.exists(p) and not os.path.exists(q): os.replace(p, q)
# ---- Pareto
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.scatter(rep.test_mean_steps, rep.test_success_pct, s=14, c="#90b0d0", edgecolors="#4a6f94", linewidths=0.4, label="Classifiers at validation-selected $p_{min}$", zorder=2)
sel = rep[rep.model == "vit_tiny"].iloc[0]
ax.scatter([sel.test_mean_steps], [sel.test_success_pct], s=26, c="#c03020", marker="D", label="Selected on validation (ViT-Tiny)", zorder=4)
ax.scatter([g.n_steps.mean()], [100.0], s=40, c="k", marker="*", label="GSS-only", zorder=4)
sty = {"focusnet_nano": ("-", "o", "#1060a0"), "yolo26n": ("--", "s", "#208050"), "convnext_tiny": ("-.", "^", "#e0a060"), "resnet18": (":", "v", "#7a4fa0")}
for m, (ls, mk, col) in sty.items():
    q = sw[sw.model == m].sort_values("p_min")
    ax.plot(q.mean_steps, q.success_pct, ls, marker=mk, ms=2.8, lw=0.9, c=col, label=f"{NAME[m]}, $p_{{min}}$ sweep", zorder=3)
ax.set_xlabel("Mean search steps"); ax.set_ylabel("Success rate (%)"); ax.set_ylim(83, 101.5); ax.set_xlim(4, 13.6)
ax.grid(alpha=0.25, lw=0.4); ax.legend(loc="lower right", frameon=True, framealpha=0.9, borderpad=0.4, handlelength=2.2)
fig.tight_layout(pad=0.4); fig.savefig(OUT + "fig_pareto.png", dpi=300); plt.close(fig)
# ---- Step distribution
order = rep.sort_values("test_mean_steps").model.tolist()
data = [tp[tp.model == m].n_steps.values for m in order] + [g.n_steps.values]
labels = [NAME[m] for m in order] + ["GSS-only"]
fig, ax = plt.subplots(figsize=(3.5, 3.1))
bp = ax.boxplot(data[::-1], vert=False, whis=1.5, widths=0.6, patch_artist=True, flierprops=dict(marker="o", ms=1.8, mfc="none", mec="0.3", mew=0.5), medianprops=dict(color="k", lw=0.9), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
for i, b in enumerate(bp["boxes"]): b.set_facecolor("#e0a060" if i == 0 else "#90b0d0")
ax.set_yticks(range(1, len(labels) + 1)); ax.set_yticklabels(labels[::-1]); ax.set_xlabel("Search steps per FOV"); ax.grid(axis="x", alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.4); fig.savefig(OUT + "fig_steps.png", dpi=300); plt.close(fig)
# ---- Validation
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.8), sharex=True)
hl = {"vit_tiny": "-", "focusnet_nano": "--", "focusnet_micro": ":"}; hc = {"vit_tiny": "#c03020", "focusnet_nano": "#1060a0", "focusnet_micro": "#208050"}
for m in NAME:
    q = vg[vg.model == m].sort_values("p_min"); k = m in hl
    kw = dict(c=hc[m], ls=hl[m], lw=1.1, label=NAME[m], zorder=3) if k else dict(c="0.7", lw=0.6, zorder=1)
    a1.plot(q.p_min, q.n_success, **kw); a2.plot(q.p_min, q.mean_steps, **kw)
a1.axhline(62, c="k", lw=0.5, ls=(0, (1, 2))); a1.text(0.505, 62.08, "GSS-only (62)", fontsize=5.5, va="bottom")
a1.set_ylabel("Validation successes (of 63)"); a2.set_ylabel("Mean validation steps"); a2.set_xlabel("Confidence threshold $p_{min}$")
a1.legend(loc="lower right", frameon=True, framealpha=0.9, handlelength=2.2)
for a in (a1, a2): a.grid(alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.4); fig.savefig(OUT + "fig_val.png", dpi=300); plt.close(fig)
print("figures done")
