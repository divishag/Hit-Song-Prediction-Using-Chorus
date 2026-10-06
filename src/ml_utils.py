import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


DATA_FILE = "data/features/features_548.csv"

RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

def get_test_metadata():

    df = pd.read_csv(DATA_FILE)

    X = df[
        [f"f{i}" for i in range(1, 519)]
    ].values

    y = df["hit"].values

    indices = np.arange(len(df))

    (
        _,
        _,
        _,
        _,
        _,
        test_indices
    ) = train_test_split(
        X,
        y,
        indices,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    return df.iloc[test_indices][
        ["artist", "song"]
    ].reset_index(drop=True)

def get_data():

    X_train, X_test, y_train, y_test = get_raw_data()

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

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
    print("DATA PREPARATION")
    print("========================================")

    print("Full dataset:", X.shape)

    print(
        "Hits:",
        np.sum(y == 1)
    )

    print(
        "Non-hits:",
        np.sum(y == 0)
    )

    print(
        "Train:",
        X_train.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    print(
        "PCA components:",
        X_train_pca.shape[1]
    )

    print(
        "Explained variance:",
        pca.explained_variance_ratio_.sum()
    )

    return (
        X_train_pca,
        X_test_pca,
        y_train,
        y_test,
        pca
    )


def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    pca
):

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    if hasattr(model, "predict_proba"):

        y_score = model.predict_proba(
            X_test
        )[:, 1]

    elif hasattr(model, "decision_function"):

        y_score = model.decision_function(
            X_test
        )

    else:

        y_score = None


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


    if y_score is not None:

        roc_auc = roc_auc_score(
            y_test,
            y_score
        )

    else:

        roc_auc = np.nan


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


    result_row = {

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

        "pca_components": X_train.shape[1],

        "pca_variance": (
            pca.explained_variance_ratio_.sum()
        )
    }


    if METRICS_FILE.exists():

        metrics_df = pd.read_csv(
            METRICS_FILE
        )

        metrics_df = metrics_df[
            metrics_df["model"] != name
        ]

    else:

        metrics_df = pd.DataFrame()


    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame(
                [result_row]
            )
        ],
        ignore_index=True
    )


    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )


    print(
        "\nSaved metrics to:",
        METRICS_FILE
    )

def get_raw_data():

    df = pd.read_csv(DATA_FILE)

    feature_cols = [
        f"f{i}"
        for i in range(1, 519)
    ]

    X = df[feature_cols].values
    y = df["hit"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,  #preserve same class distribution in train and test sets
        random_state=42 #reproducibility
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )