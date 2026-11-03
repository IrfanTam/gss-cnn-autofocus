"""
Data preparation module for the hybrid GSS-CNN autofocus pipeline (MethodsX).

Scope and decisions are pinned in ../../PROJECT.md -- do not change labeling
scheme, excluded FOV list, or dataset sources here without updating that file
first.

Pipeline:
    1. Load all 5 sub-datasets from FanLAB/Dataset Master, normalizing each
       to a common schema (dataset, fov_id, pos, brenner, filepath).
    2. Drop the 10 manually-confirmed anomalous FOV (excluded_fov.json).
    3. Label each image with P_i = brenner / brenner_max(FOV), 3 classes:
       Focused (P_i > 0.55), Medium (0.30 < P_i <= 0.55), Blur (P_i <= 0.30).
    4. Split FOV-level 70/15/15 (train/val/test), stratified per source
       dataset so no single dataset dominates a split.

Run directly to regenerate the final CSVs:
    python data_preparation.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_MASTER = Path(r"D:\2026\Muhammad Irfan Tam Tomo\PENS S2\FanLAB\Dataset Master")
EXCLUDED_FOV_PATH = Path(r"D:\2026\Muhammad Irfan Tam Tomo\PENS S2\Bu Heny\Analisa Dataset\excluded_fov.json")
OUTPUT_DIR = Path(__file__).parent / "output"

# P_i thresholds -- pinned in PROJECT.md, do not change without updating it.
P_FOCUSED = 0.55
P_MEDIUM = 0.30
CLASS_NAMES = ["Focused", "Medium", "Blur"]

# Each entry: dataset folder name -> config describing how to read its CSV
# and how to derive (fov_id, pos, brenner, filepath) from it.
DATASETS = {
    "Dataset_100FOV_101img": dict(
        csv="data.csv", fov_col="lapang_pandang", pos_col="sudut",
        brenner_col="brenner", path_col="filepath",
    ),
    "Dataset_100FOV_401img": dict(
        csv="data.csv", fov_col="lapang_pandang", pos_col="sudut",
        brenner_col="brenner", path_col="filepath",
    ),
    "Dataset_49FOV_90img": dict(
        csv="data.csv", fov_col="lapang_pandang", pos_col="sudut",
        brenner_col="brenner", path_col="filepath",
    ),
    "Dataset_99FOV_120img": dict(
        csv="data.csv", fov_col="lapang_pandang", pos_col="sudut",
        brenner_col="brenner", path_col="filepath",
    ),
    "Dataset_93FOV_101img_1incomplete": dict(
        csv="dataset_metadata.csv", fov_col=None, pos_col="posisi_aktual",
        brenner_col="brenner", path_col="image_filename",
    ),
}


def load_excluded_fov():
    """Load the manually-confirmed anomalous FOV list (dataset -> [fov_id, ...])."""
    with open(EXCLUDED_FOV_PATH) as f:
        return json.load(f)


def load_one_dataset(name):
    """Load & normalize a single dataset -> DataFrame with standard columns."""
    cfg = DATASETS[name]
    df = pd.read_csv(DATASET_MASTER / name / cfg["csv"])

    if cfg["fov_col"] is not None:
        fov_id = df[cfg["fov_col"]].astype(str)
    else:
        # FOV id is embedded in the image path, e.g. "Lapang_Pandang_1/sample_pos_431.jpg"
        fov_id = df[cfg["path_col"]].str.extract(r"(Lapang_Pandang_\d+|LapangPandang_\d+)", expand=False)

    out = pd.DataFrame({
        "dataset": name,
        "fov_id": fov_id,
        "pos": df[cfg["pos_col"]].astype(float),
        "brenner": df[cfg["brenner_col"]].astype(float),
        "filepath": df[cfg["path_col"]],
    })
    return out.dropna(subset=["fov_id", "pos", "brenner", "filepath"])


def load_all_datasets():
    """Load and concatenate all 5 sub-datasets into one long DataFrame."""
    frames = [load_one_dataset(name) for name in DATASETS]
    combined = pd.concat(frames, ignore_index=True)
    combined["abs_filepath"] = combined.apply(
        lambda r: str(DATASET_MASTER / r["dataset"] / r["filepath"].replace("\\", "/")), axis=1
    )
    return combined


def exclude_anomalous_fov(df):
    """Drop rows belonging to the 10 manually-confirmed anomalous FOV."""
    excluded = load_excluded_fov()
    mask = pd.Series(False, index=df.index)
    for dataset, fov_ids in excluded.items():
        mask |= (df["dataset"] == dataset) & (df["fov_id"].isin(fov_ids))
    removed = int(mask.sum())
    print(f"Excluding {removed} rows from {mask.groupby(df['dataset']).sum().sum()} "
          f"anomalous FOV across {sum(len(v) for v in excluded.values())} FOV entries.")
    return df[~mask].reset_index(drop=True)


def label_p_i(df):
    """
    Add P_i (Brenner normalized to per-FOV peak) and 3-class label.
    Class = Focused if P_i > 0.55, Medium if 0.30 < P_i <= 0.55, else Blur.
    """
    df = df.copy()
    peak = df.groupby(["dataset", "fov_id"])["brenner"].transform("max")
    df["p_i"] = df["brenner"] / peak

    conditions = [df["p_i"] > P_FOCUSED, df["p_i"] > P_MEDIUM]
    choices = ["Focused", "Medium"]
    df["focus_class"] = np.select(conditions, choices, default="Blur")
    return df


def balance_classes(df, random_state=42):
    """
    Downsample ALL classes to the size of the smallest class (Medium, ~2,135
    images), applied BEFORE the FOV-level split so every split (train/val/
    test) ends up ~1:1:1 automatically.

    IMPORTANT trade-off (see PROJECT.md decision log): this shrinks the
    usable dataset from 75,125 to ~6,405 images total (~8.5%), since Focused
    and Medium only have ~2,135-2,486 images in the ENTIRE dataset -- there
    is no way to "find more" of them, this is the real ceiling. Val/test
    are also balanced under this scheme, which means test-set metrics
    (recall/precision per class) STOP representing the real-world class
    distribution (~94% Blur in actual deployment). This is a deliberate
    evaluation-design choice, not a bug -- must be stated explicitly in the
    paper's Method Validation section so reviewers don't mistake the
    balanced-test-set numbers for real-world deployment performance.
    """
    counts = df["focus_class"].value_counts()
    min_class, min_n = counts.idxmin(), counts.min()
    print(f"Balancing all classes to {min_n:,} (smallest class: {min_class}) "
          f"-- dataset shrinks from {len(df):,} to ~{min_n * len(counts):,} images.")

    parts = []
    for cls in CLASS_NAMES:
        cls_df = df[df["focus_class"] == cls]
        if len(cls_df) > min_n:
            cls_df = cls_df.sample(min_n, random_state=random_state)
        parts.append(cls_df)

    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)


def split_by_fov(df, train_ratio=0.70, val_ratio=0.15, random_state=42):
    """
    FOV-level split, stratified per source dataset so no single dataset
    dominates any split and no FOV leaks across splits.
    """
    train_parts, val_parts, test_parts = [], [], []

    for dataset_name, g in df.groupby("dataset"):
        fov_ids = g["fov_id"].unique()
        test_ratio = 1.0 - train_ratio - val_ratio

        train_ids, temp_ids = train_test_split(
            fov_ids, test_size=(val_ratio + test_ratio), random_state=random_state
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_state
        )

        train_parts.append(g[g["fov_id"].isin(train_ids)])
        val_parts.append(g[g["fov_id"].isin(val_ids)])
        test_parts.append(g[g["fov_id"].isin(test_ids)])

        print(f"  {dataset_name}: {len(train_ids)} train / {len(val_ids)} val / "
              f"{len(test_ids)} test FOV")

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df["split"] = name

    return train_df, val_df, test_df


def print_class_distribution(df, label):
    print(f"\n{label} class distribution ({len(df)} images):")
    counts = df["focus_class"].value_counts()
    for cls in CLASS_NAMES:
        n = counts.get(cls, 0)
        print(f"  {cls:8s}: {n:6d} ({n / len(df) * 100:5.1f}%)")


def build_dataset(save=True, balanced=False):
    """
    Full pipeline: load -> exclude anomalies -> label -> (optionally balance
    1:1:1) -> split -> (optionally) save.

    balanced=False (default): original ~94% Blur / 3% Medium / 3% Focused
        proportions preserved in every split -- reflects real-world
        deployment distribution. Output files: train.csv / val.csv / test.csv
    balanced=True: all classes downsampled to the smallest class count
        BEFORE splitting, so train/val/test are all ~1:1:1. Output files
        get a "_balanced" suffix so both versions can coexist and be
        compared -- see PROJECT.md for why this is a deliberate evaluation-
        design trade-off, not a strict improvement.
    """
    print("Loading all datasets...")
    df = load_all_datasets()
    print(f"Loaded {len(df)} rows from {df['dataset'].nunique()} datasets, "
          f"{df.groupby('dataset')['fov_id'].nunique().sum()} FOV total.")

    df = exclude_anomalous_fov(df)
    df = label_p_i(df)
    print_class_distribution(df, "Overall (before balancing)")

    if balanced:
        df = balance_classes(df)
        print_class_distribution(df, "Overall (after 1:1:1 balancing)")

    print("\nSplitting FOV-level (70/15/15, stratified per dataset)...")
    train_df, val_df, test_df = split_by_fov(df)

    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print_class_distribution(split_df, name)

    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        suffix = "_balanced" if balanced else ""
        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        full_df.to_csv(OUTPUT_DIR / f"labeled_dataset{suffix}.csv", index=False)
        train_df.to_csv(OUTPUT_DIR / f"train{suffix}.csv", index=False)
        val_df.to_csv(OUTPUT_DIR / f"val{suffix}.csv", index=False)
        test_df.to_csv(OUTPUT_DIR / f"test{suffix}.csv", index=False)
        print(f"\nSaved -> {OUTPUT_DIR}/")
        print(f"  labeled_dataset{suffix}.csv  (all splits combined, with 'split' column)")
        print(f"  train{suffix}.csv / val{suffix}.csv / test{suffix}.csv")

    return train_df, val_df, test_df


if __name__ == "__main__":
    import sys
    balanced = "--balanced" in sys.argv
    build_dataset(balanced=balanced)
