"""
Unimodality check for the 71 test-split FOV used in Tahap 3 simulation --
verifies the assumption GSS depends on (Brenner curve has a single
dominant peak per FOV). Reuses simulation_fov_data.csv directly (already
has the full, leak-free Brenner curve per FOV) rather than reloading from
Dataset Master.

Criterion (same as Analisa Dataset/brenner_anomaly_check.ipynb): a FOV is
unimodal if its Brenner curve has at most 1 local maximum with amplitude
> 20% of the global peak; secondary peaks below that are treated as noise.

Usage:
    python unimodality_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

DATA_PATH = Path(__file__).parent.parent / "03_pipeline_validation" / "output" / "simulation_fov_data.csv"
OUTPUT_DIR = Path(__file__).parent / "output"

PEAK_RATIO_THRESH = 0.20
LOCAL_MAX_ORDER = 2


def check_unimodal(brenner, order=LOCAL_MAX_ORDER, peak_ratio_thresh=PEAK_RATIO_THRESH):
    arr = np.asarray(brenner)
    global_max = arr.max()

    local_max_idx = argrelextrema(arr, np.greater, order=order)[0]
    if len(local_max_idx) == 0:
        peak_idx = int(np.argmax(arr))
        return True, 1, peak_idx

    significant_peaks = [i for i in local_max_idx if arr[i] >= peak_ratio_thresh * global_max]
    n_peaks = len(significant_peaks)
    peak_idx = int(np.argmax(arr))
    return n_peaks <= 1, max(n_peaks, 1), peak_idx


def analyze_all(df):
    rows = []
    for (dataset, fov_id), g in df.groupby(["dataset", "fov_id"]):
        g = g.sort_values("pos")
        brenner = g["brenner"].values
        pos = g["pos"].values

        is_uni, n_peaks, peak_idx = check_unimodal(brenner)
        snr = brenner[peak_idx] / (brenner.mean() + 1e-8)

        rows.append({
            "dataset": dataset, "fov_id": fov_id,
            "n_points": len(brenner), "is_unimodal": is_uni, "n_peaks": n_peaks,
            "peak_pos": float(pos[peak_idx]), "peak_brenner": float(brenner[peak_idx]),
            "snr": float(snr),
        })
    return pd.DataFrame(rows)


def plot_representative_curves(df, results_df, out_path, n=5):
    """Same style as the original Fig. 4 in access_final.tex / Analisa Dataset notebook."""
    unimodal = results_df[results_df["is_unimodal"]].sort_values("snr")
    non_unimodal = results_df[~results_df["is_unimodal"]]

    selected = []
    if len(unimodal) > 0:
        idxs = np.linspace(0, len(unimodal) - 1, min(n - 1, len(unimodal)), dtype=int)
        selected.extend(unimodal.iloc[idxs].to_dict("records"))
    if len(non_unimodal) > 0:
        selected.append(non_unimodal.iloc[0].to_dict())
    selected = selected[:n]

    fig, axes = plt.subplots(1, len(selected), figsize=(3.2 * len(selected), 3.5))
    if len(selected) == 1:
        axes = [axes]
    colors = ["#2c7bb6", "#1a9641", "#d7191c", "#756bb1", "#e6550d"]

    for i, (ax, sel) in enumerate(zip(axes, selected)):
        g = df[(df["dataset"] == sel["dataset"]) & (df["fov_id"] == sel["fov_id"])].sort_values("pos")
        brenner = g["brenner"].values
        pos = g["pos"].values
        b_norm = (brenner - brenner.min()) / (brenner.max() - brenner.min() + 1e-8)

        ax.plot(pos, b_norm, color=colors[i % len(colors)], linewidth=2, marker="o", markersize=2)
        peak_idx = int(np.argmax(brenner))
        ax.axvline(pos[peak_idx], color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"$z^*$={pos[peak_idx]:.1f}")
        ax.scatter([pos[peak_idx]], [b_norm[peak_idx]], color="red", zorder=5, s=50)

        label = "Unimodal" if sel["is_unimodal"] else "Non-unimodal"
        ax.set_title(f"FOV {i+1}\n({label}, SNR={sel['snr']:.1f})", fontsize=9)
        ax.set_xlabel("Motor Position (deg)", fontsize=8)
        if i == 0:
            ax.set_ylabel("Normalized Brenner Score", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    plt.suptitle("Representative Brenner Gradient Curves (Test-Split FOV, n=71)", fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["fov_id"] = df["fov_id"].astype(str)
    print(f"Loaded {len(df)} rows, {df.groupby('dataset')['fov_id'].nunique().sum()} FOV.")

    results_df = analyze_all(df)
    n_uni = results_df["is_unimodal"].sum()
    total = len(results_df)

    print(f"\nUnimodal: {n_uni}/{total} ({n_uni/total*100:.1f}%)")
    print(f"Non-unimodal: {total-n_uni}/{total} ({(total-n_uni)/total*100:.1f}%)")
    print(f"Peak position range: {results_df['peak_pos'].min():.1f}-{results_df['peak_pos'].max():.1f} deg")
    print(f"Mean SNR: {results_df['snr'].mean():.2f} +/- {results_df['snr'].std():.2f}")

    print("\nPer-dataset breakdown:")
    print(results_df.groupby("dataset").agg(
        n_fov=("fov_id", "count"), n_unimodal=("is_unimodal", "sum")
    ).assign(pct=lambda d: (d["n_unimodal"] / d["n_fov"] * 100).round(1)))

    results_df.to_csv(OUTPUT_DIR / "unimodality_check.csv", index=False)
    print(f"\nSaved -> {OUTPUT_DIR / 'unimodality_check.csv'}")

    plot_representative_curves(df, results_df, OUTPUT_DIR / "brenner_curves_test_split.png")
    print(f"Saved -> {OUTPUT_DIR / 'brenner_curves_test_split.png'}")


if __name__ == "__main__":
    main()
