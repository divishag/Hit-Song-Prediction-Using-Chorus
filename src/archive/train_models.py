import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ==========================================================
# PATHS
# ==========================================================

DATA_FILE = "data/features/features_548.csv"

RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = RESULTS_DIR / "model_metrics.csv"


# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_FILE)

feature_cols = [
    f"f{i}"
    for i in range(1, 519)
]

X = df[feature_cols].values
y = df["hit"].values


print("\n========================================")
print("DATASET")
print("========================================")

print("Full dataset:", X.shape)
print("Hits:", np.sum(y == 1))
print("Non-hits:", np.sum(y == 0))


# ==========================================================
# 2. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)


print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print("Train:", X_train.shape)
print("Test:", X_test.shape)

print("Train hits:", np.sum(y_train == 1))
print("Train non-hits:", np.sum(y_train == 0))

print("Test hits:", np.sum(y_test == 1))
print("Test non-hits:", np.sum(y_test == 0))


# ==========================================================
# 3. STANDARDIZATION
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================================
# 4. PCA
# ==========================================================

pca = PCA(
    n_components=0.95
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)


print("\n========================================")
print("PCA")
print("========================================")

print(
    "Original features:",
    X_train.shape[1]
)

print(
    "PCA components:",
    X_train_pca.shape[1]
)

print(
    "Explained variance:",
    pca.explained_variance_ratio_.sum()
)


# ==========================================================
# 5. STORE RESULTS
# ==========================================================

results = []


# ==========================================================
# 6. MODEL EVALUATION
# ==========================================================

def evaluate_model(name, model):

    model.fit(
        X_train_pca,
        y_train
    )

    y_pred = model.predict(
        X_test_pca
    )

    y_prob = model.predict_proba(
        X_test_pca
    )[:, 1]


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )


    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()


    print("\n========================================")
    print(name)
    print("========================================")

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Balanced Accuracy:",
        round(balanced_accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "F1:",
        round(f1, 4)
    )

    print(
        "ROC-AUC:",
        round(roc_auc, 4)
    )


    print("\nConfusion Matrix:")
    print(cm)


    # Save result
    results.append({

        "model": name,

        "accuracy": accuracy,

        "balanced_accuracy": balanced_accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc,

        "true_negative": tn,

        "false_positive": fp,

        "false_negative": fn,

        "true_positive": tp,

        "pca_components": X_train_pca.shape[1],

        "pca_variance": (
            pca.explained_variance_ratio_.sum()
        )

    })


# ==========================================================
# 7. NORMAL LOGISTIC REGRESSION
# ==========================================================

normal_lr = LogisticRegression(
    max_iter=5000,
    random_state=42
)

evaluate_model(
    "Logistic Regression",
    normal_lr
)


# ==========================================================
# 8. BALANCED LOGISTIC REGRESSION
# ==========================================================

balanced_lr = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=42
)

evaluate_model(
    "Logistic Regression Balanced",
    balanced_lr
)


# ==========================================================
# 9. SAVE METRICS CSV
# ==========================================================

metrics_df = pd.DataFrame(
    results
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


print("\n========================================")
print("RESULTS SAVED")
print("========================================")

print(metrics_df)

print(
    "\nSaved to:",
    METRICS_FILE
)