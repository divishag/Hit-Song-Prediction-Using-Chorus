from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
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

from ml_utils import get_raw_data


# ==========================================================
# DATA
# ==========================================================

X_train, X_test, y_train, y_test = get_raw_data()


# ==========================================================
# PIPELINE
# Scaling + PCA are fitted separately inside every CV fold
# ==========================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    (
        "model",
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        )
    )
])


# ==========================================================
# PARAMETER GRID
# ==========================================================

param_grid = {
    "model__n_estimators": [100, 200, 500],

    "model__max_depth": [
        None,
        5,
        10,
        20
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__max_features": [
        "sqrt",
        "log2"
    ]
}


# ==========================================================
# 5-FOLD CV
# ==========================================================

print("\n========================================")
print("RANDOM FOREST LEAKAGE-FREE 5-FOLD CV")
print("========================================")


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=1
)


grid_search.fit(
    X_train,
    y_train
)


# ==========================================================
# BEST PARAMETERS
# ==========================================================

print("\n========================================")
print("BEST RANDOM FOREST PARAMETERS")
print("========================================")

print(
    grid_search.best_params_
)

print(
    "Best CV balanced accuracy:",
    round(grid_search.best_score_, 4)
)


# ==========================================================
# FINAL TEST
# ==========================================================

best_pipeline = grid_search.best_estimator_

y_pred = best_pipeline.predict(
    X_test
)

y_prob = best_pipeline.predict_proba(
    X_test
)[:, 1]


print("\n========================================")
print("RANDOM FOREST FINAL TEST")
print("========================================")

print(
    "Accuracy:",
    round(
        accuracy_score(y_test, y_pred),
        4
    )
)

print(
    "Balanced Accuracy:",
    round(
        balanced_accuracy_score(
            y_test,
            y_pred
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(y_test, y_pred),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(y_test, y_pred),
        4
    )
)

print(
    "F1:",
    round(
        f1_score(y_test, y_pred),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            y_prob
        ),
        4
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print(
    "\nPCA components in final fitted pipeline:",
    best_pipeline["pca"].n_components_
)