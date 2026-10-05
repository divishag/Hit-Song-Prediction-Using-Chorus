from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path("data/features/features_548.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "lda_hit_predictor.joblib"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_FILE}. Run this script from the project root "
        "after feature extraction has created features_548.csv."
    )

df = pd.read_csv(DATA_FILE)
feature_cols = [f"f{i}" for i in range(1, 519)]

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Feature CSV is missing columns, starting with: {missing[:5]}")

X = df[feature_cols].values
y = df["hit"].values

# Deployment model: after LDA was selected using the held-out evaluation,
# refit the same pipeline on all 548 labelled songs for the live demo.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("model", LinearDiscriminantAnalysis()),
])

model.fit(X, y)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_FILE)

print("========================================")
print("DEMO MODEL SAVED")
print("========================================")
print("Songs used:", len(df))
print("Features:", len(feature_cols))
print("PCA components:", model["pca"].n_components_)
print("Saved to:", MODEL_FILE)
