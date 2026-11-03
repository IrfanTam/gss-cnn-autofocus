"""
Step A of the validation-based selection protocol.

Extract FULL Brenner curves (every scanned position) for the FOV in the
classifier's VALIDATION split (63 FOV). These FOV are disjoint from train and
test (checked below), so they can be used to choose the backbone and the
confidence threshold p_min WITHOUT touching the test FOV that are reported.

Output: output_val/simulation_fov_data_val.csv  (same columns as
output/simulation_fov_data.csv). Run in the environment that can see the
Dataset Master folder (same as prepare_simulation_data.py).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "01_data_preparation"))
from data_preparation import load_all_datasets, exclude_anomalous_fov, label_p_i  # noqa: E402

PREP_OUT = Path(__file__).parent.parent / "01_data_preparation" / "output"
OUT_DIR = Path(__file__).parent / "output_val"


def fov_set(name):
    d = pd.read_csv(PREP_OUT / f"{name}_balanced.csv")
    return set(zip(d["dataset"], d["fov_id"].astype(str)))


def main():
    val, test, train = fov_set("val"), fov_set("test"), fov_set("train")
    assert not (val & test), "val/test FOV overlap -- selection would leak into the test report"
    assert not (val & train), "val/train FOV overlap"
    print(f"val FOV: {len(val)} (test {len(test)}, train {len(train)}); all disjoint")

    df = label_p_i(exclude_anomalous_fov(load_all_datasets()))
    df["fov_id"] = df["fov_id"].astype(str)
    mask = df.apply(lambda r: (r["dataset"], r["fov_id"]) in val, axis=1)
    sim = df[mask].reset_index(drop=True)
    print(f"Extracted {len(sim)} rows across {sim.groupby('dataset')['fov_id'].nunique().sum()} FOV")
    for ds, g in sim.groupby("dataset"):
        print(f"  {ds}: {g['fov_id'].nunique()} FOV, avg {len(g) / g['fov_id'].nunique():.0f} points/FOV")
    OUT_DIR.mkdir(exist_ok=True)
    sim.to_csv(OUT_DIR / "simulation_fov_data_val.csv", index=False)
    print("Saved ->", OUT_DIR / "simulation_fov_data_val.csv")


if __name__ == "__main__":
    main()
