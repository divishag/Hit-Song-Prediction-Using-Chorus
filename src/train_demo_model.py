from pathlib import Path

import joblib

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_utils import get_raw_data


# ==========================================================
# PATHS
# ==========================================================

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "lda_hit_predictor.joblib"


# ==========================================================
# DATA
# Use the exact same fixed 75/25 split as evaluation
# ==========================================================

X_train, X_test, y_train, y_test = get_raw_data()



model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("model", LinearDiscriminantAnalysis())
])


# Train only on training data
model.fit(
    X_train,
    y_train
)



MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)


# ==========================================================
# SUMMARY
# ==========================================================

print("========================================")
print("DEMO MODEL SAVED")
print("========================================")

print(
    "Training songs used:",
    len(X_train)
)

print(
    "Held-out test songs:",
    len(X_test)
)

print(
    "Input features:",
    X_train.shape[1]
)

print(
    "PCA components:",
    model["pca"].n_components_
)

print(
    "Saved to:",
    MODEL_FILE
)