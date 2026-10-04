import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier

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

METRICS_DIR = Path("results/metrics")
PREDICTIONS_DIR = Path("results/predictions")

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_FILE)

feature_cols = [
    f"f{i}"
    for i in range(1, 519)
]

X = df[feature_cols].values
y = df["hit"].values

indices = np.arange(len(df))


(
    X_train,
    X_test,
    y_train,
    y_test,
    train_indices,
    test_indices
) = train_test_split(
    X,
    y,
    indices,
    test_size=0.25,
    stratify=y,
    random_state=42
)


test_metadata = df.iloc[test_indices][
    ["artist", "song"]
].reset_index(drop=True)


# ==========================================================
# COMMON PIPELINE
# ==========================================================

def make_pipeline(model):

    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("model", model)
    ])


# ==========================================================
# FINAL MODELS
# ==========================================================

models = {

    "Logistic Regression":
        make_pipeline(
            LogisticRegression(
                max_iter=5000,
                random_state=42
            )
        ),

    "Logistic Regression Balanced":
        make_pipeline(
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42
            )
        ),

    "Linear Discriminant Analysis":
        make_pipeline(
            LinearDiscriminantAnalysis()
        ),

    "SVM":
        make_pipeline(
            SVC(
                C=0.01,
                kernel="linear",
                random_state=42
            )
        ),

    "Random Forest Balanced":
        make_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                max_features="log2",
                min_samples_split=5,
                class_weight="balanced",
                random_state=42
            )
        ),

    # Keep this only as the imbalance comparison experiment
    "Random Forest Unbalanced":
        make_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                max_features="log2",
                min_samples_split=5,
                random_state=42
            )
        ),

    "Gradient Boosting":
        make_pipeline(
            GradientBoostingClassifier(
                learning_rate=0.1,
                max_depth=3,
                n_estimators=200,
                subsample=1.0,
                random_state=42
            )
        ),

    "Neural Network":
        make_pipeline(
            MLPClassifier(
                activation="tanh",
                alpha=0.001,
                hidden_layer_sizes=(32,),
                learning_rate_init=0.001,
                max_iter=2000,
                early_stopping=False,
                random_state=42
            )
        )
}


# ==========================================================
# RUN MODELS
# ==========================================================

results = []


for name, model in models.items():

    print("\n========================================")
    print(name)
    print("========================================")

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )


    if hasattr(
        model["model"],
        "predict_proba"
    ):

        y_score = model.predict_proba(
            X_test
        )[:, 1]

    else:

        y_score = model.decision_function(
            X_test
        )


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
        y_score
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()


    print("Accuracy:", round(accuracy, 4))
    print("Balanced Accuracy:", round(balanced_accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(cm)


    # ------------------------------------------------------
    # SAVE METRICS
    # ------------------------------------------------------

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

        "pca_components":
            model["pca"].n_components_,

        "pca_variance":
            model["pca"]
            .explained_variance_ratio_
            .sum()
    })


    # ------------------------------------------------------
    # SAVE PREDICTIONS
    # ------------------------------------------------------

    prediction_df = test_metadata.copy()

    prediction_df["actual"] = y_test
    prediction_df["predicted"] = y_pred
    prediction_df["score"] = y_score

    prediction_df["correct"] = (
        prediction_df["actual"]
        ==
        prediction_df["predicted"]
    )


    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_predictions.csv"
    )


    prediction_df.to_csv(
        PREDICTIONS_DIR / filename,
        index=False
    )


# ==========================================================
# SAVE FINAL METRICS
# ==========================================================

metrics_df = pd.DataFrame(
    results
)

metrics_df.to_csv(
    METRICS_DIR / "model_metrics.csv",
    index=False
)


# ==========================================================
# SAVE FINAL BEST PARAMETERS
# ==========================================================

best_params = pd.DataFrame([

    {
        "model": "SVM",
        "best_parameters":
            "C=0.01, kernel=linear",
        "cv_metric":
            "balanced_accuracy",
        "best_cv_score":
            0.6515
    },

    {
        "model": "Random Forest Balanced",
        "best_parameters":
            "max_depth=5, "
            "max_features=log2, "
            "min_samples_split=5, "
            "n_estimators=200, "
            "class_weight=balanced",
        "cv_metric":
            "balanced_accuracy",
        "best_cv_score":
            0.6551
    },

    {
        "model": "Gradient Boosting",
        "best_parameters":
            "learning_rate=0.1, "
            "max_depth=3, "
            "n_estimators=200, "
            "subsample=1.0",
        "cv_metric":
            "balanced_accuracy",
        "best_cv_score":
            0.6048
    },

    {
        "model": "Neural Network",
        "best_parameters":
            "activation=tanh, "
            "alpha=0.001, "
            "hidden_layer_sizes=(32,), "
            "learning_rate_init=0.001, "
            "early_stopping=False",
        "cv_metric":
            "balanced_accuracy",
        "best_cv_score":
            0.6405
    }

])


best_params.to_csv(
    METRICS_DIR / "best_params.csv",
    index=False
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n========================================")
print("FINAL RESULTS SAVED")
print("========================================")

print(
    metrics_df[
        [
            "model",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "roc_auc"
        ]
    ].sort_values(
        "balanced_accuracy",
        ascending=False
    ).to_string(index=False)
)

print(
    "\nMetrics:",
    METRICS_DIR / "model_metrics.csv"
)

print(
    "Best parameters:",
    METRICS_DIR / "best_params.csv"
)

print(
    "Predictions:",
    PREDICTIONS_DIR
)