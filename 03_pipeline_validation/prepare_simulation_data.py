"""
Extract FULL Brenner curves (every scanned position, not the class-balanced
subset) for the FOV that fall in the classifier's TEST split -- these are
the only FOV the CNN never saw a single image from during training, so
using them for the autofocus pipeline simulation is leak-free.

Why not the balanced dataset (train/val/test_balanced.csv)? Per-image
downsampling there removes random positions from the middle of each FOV's
curve. GSS/Hill Climbing/etc. need to be able to "visit" ANY position a real
scan would have covered -- so this script re-reads the FULL scan (101/120/401
points per FOV, depending on source dataset) straight from Dataset Master,
filtered to only the 71 FOV that are in test_balanced.csv's (dataset, fov_id)
pairs. See PROJECT.md ("PENTING -- sifat tahap ini SIMULASI...") for the
full rationale.

Output: simulation_fov_data.csv -- one row per (FOV, scanned position),
columns: dataset, fov_id, pos, brenner, abs_filepath, p_i (for reference).
Also excludes the 10 already-known anomalous FOV (excluded_fov.json),
consistent with 01_data_preparation.
"""

import sys
from pathlib import Path

import pandas as pd

# Reuse the exact same loading/labeling logic as 01_data_preparation, rather
# than duplicating it -- keeps the P_i formula and anomaly-exclusion list
# as a single source of truth.
sys.path.insert(0, str(Path(__file__).parent.parent / "01_data_preparation"))
from data_preparation import (  # noqa: E402
    load_all_datasets, exclude_anomalous_fov, label_p_i,
)

DATA_PREP_OUTPUT = Path(__file__).parent.parent / "01_data_preparation" / "output"
OUTPUT_DIR = Path(__file__).parent / "output"


def get_test_fov_set():
    """(dataset, fov_id) pairs that are in the classifier's test split -- the
    only FOV safe to simulate on without leaking training data."""
    test_df = pd.read_csv(DATA_PREP_OUTPUT / "test_balanced.csv")
    return set(zip(test_df["dataset"], test_df["fov_id"].astype(str)))


def build_simulation_data(save=True):
    print("Loading full dataset (all scanned positions, no balancing)...")
    df = load_all_datasets()
    df = exclude_anomalous_fov(df)
    df = label_p_i(df)

    test_fov = get_test_fov_set()
    print(f"\nTest-split FOV (leak-free for simulation): {len(test_fov)}")

    df["fov_id"] = df["fov_id"].astype(str)
    mask = df.apply(lambda r: (r["dataset"], r["fov_id"]) in test_fov, axis=1)
    sim_df = df[mask].reset_index(drop=True)

    print(f"\nExtracted {len(sim_df)} rows (full scan positions) across "
          f"{sim_df.groupby('dataset')['fov_id'].nunique().sum()} FOV.")

    print("\nPoints-per-FOV (should match source dataset's full scan density -- "
          "101/120/401 etc, NOT the balanced-subset count):")
    for dataset_name, g in sim_df.groupby("dataset"):
        n_fov = g["fov_id"].nunique()
        avg_points = len(g) / n_fov
        print(f"  {dataset_name}: {n_fov} FOV, avg {avg_points:.0f} points/FOV")

    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / "simulation_fov_data.csv"
        sim_df.to_csv(out_path, index=False)
        print(f"\nSaved -> {out_path}")

    return sim_df


if __name__ == "__main__":
    build_simulation_data()
