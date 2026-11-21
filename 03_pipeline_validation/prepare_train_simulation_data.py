"""
Supplementary analysis (Step T1): extract FULL Brenner curves for the 299 TRAIN FOV.

These FOV were used to train the classifiers, so the replay results on them are
OPTIMISTIC and must be reported separately from the validation/test results.
Output: output_train/simulation_fov_data_train.csv (same columns as the val/test files).
Run in the same environment as prepare_val_simulation_data.py.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "01_data_preparation"))
from data_preparation import load_all_datasets, exclude_anomalous_fov, label_p_i  # noqa: E402

PREP_OUT = Path(__file__).parent.parent / "01_data_preparation" / "output"
OUT_DIR = Path(__file__).parent / "output_train"


def fov_set(name):
    d = pd.read_csv(PREP_OUT / f"{name}_balanced.csv")
    return set(zip(d["dataset"], d["fov_id"].astype(str)))


def main():
    train, val, test = fov_set("train"), fov_set("val"), fov_set("test")
    assert not (train & val) and not (train & test), "train overlaps val/test"
    print(f"train FOV: {len(train)} (val {len(val)}, test {len(test)}); all disjoint")
    df = label_p_i(exclude_anomalous_fov(load_all_datasets()))
    df["fov_id"] = df["fov_id"].astype(str)
    mask = df.apply(lambda r: (r["dataset"], r["fov_id"]) in train, axis=1)
    sim = df[mask].reset_index(drop=True)
    print(f"Extracted {len(sim)} rows across {sim.groupby('dataset')['fov_id'].nunique().sum()} FOV")
    OUT_DIR.mkdir(exist_ok=True)
    sim.to_csv(OUT_DIR / "simulation_fov_data_train.csv", index=False)
    print("Saved ->", OUT_DIR / "simulation_fov_data_train.csv")


if __name__ == "__main__":
    main()
