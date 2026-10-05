import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split


# ==========================================================
# PATHS
# ==========================================================

DATA_FILE = "data/features/features_548.csv"
OUTPUT_DIR = Path("data/splits")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD THE SAME 548-SONG DATASET
# ==========================================================

df = pd.read_csv(DATA_FILE)

feature_cols = [
    f"f{i}"
    for i in range(1, 519)
]

X = df[feature_cols].values
y = df["hit"].values

indices = np.arange(len(df))


# ==========================================================
# EXACT SAME SPLIT USED IN OUR ML
# ==========================================================

(
    train_indices,
    test_indices
) = train_test_split(
    indices,
    test_size=0.25,
    stratify=y,
    random_state=42
)


# ==========================================================
# CREATE MANIFESTS
# ==========================================================

train_df = df.iloc[
    train_indices
].reset_index(drop=True)

test_df = df.iloc[
    test_indices
].reset_index(drop=True)


# We don't need 518 features in these manifest files.
# Keep only information useful for identifying the songs.

train_manifest = train_df[
    ["artist", "song", "hit"]
]

test_manifest = test_df[
    ["artist", "song", "hit"]
]


# ==========================================================
# SAVE
# ==========================================================

train_manifest.to_csv(
    OUTPUT_DIR / "train_songs.csv",
    index=False
)

test_manifest.to_csv(
    OUTPUT_DIR / "test_songs.csv",
    index=False
)


# ==========================================================
# VERIFY
# ==========================================================

print("\n========================================")
print("TRAIN / TEST SPLIT SAVED")
print("========================================")

print("\nTraining songs:", len(train_manifest))

print(
    train_manifest["hit"]
    .value_counts()
    .sort_index()
)

print("\nTest songs:", len(test_manifest))

print(
    test_manifest["hit"]
    .value_counts()
    .sort_index()
)

print(
    "\nTotal:",
    len(train_manifest)
    + len(test_manifest)
)

print(
    "\nOverlap:",
    len(
        set(train_indices)
        &
        set(test_indices)
    )
)

print(
    "\nSaved:"
)

print(
    OUTPUT_DIR / "train_songs.csv"
)

print(
    OUTPUT_DIR / "test_songs.csv"
)