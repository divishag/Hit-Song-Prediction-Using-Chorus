from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ml_utils import get_raw_data


# ==========================================================
# DATA
# ==========================================================

X_train, X_test, y_train, y_test = get_raw_data()


# ==========================================================
# PIPELINE
# ==========================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("model", SVC())
])


# ==========================================================
# PARAMETER GRID
# ==========================================================

param_grid = [
    {
        "model__kernel": ["linear"],
        "model__C": [0.01, 0.1, 1, 10]
    },

    {
        "model__kernel": ["rbf"],
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": [
            "scale",
            0.001,
            0.01,
            0.1
        ]
    },

    {
        "model__kernel": ["poly"],
        "model__C": [0.1, 1, 10],
        "model__degree": [2, 3],
        "model__gamma": ["scale"]
    }
]


# ==========================================================
# 5-FOLD CV
# ==========================================================

print("\n========================================")
print("SVM LEAKAGE-FREE 5-FOLD CV")
print("========================================")


grid_search = GridSearchCV(
    pipeline,
    param_grid,
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
# RESULTS
# ==========================================================

print("\n========================================")
print("BEST SVM PARAMETERS")
print("========================================")

print(
    grid_search.best_params_
)

print(
    "Best CV balanced accuracy:",
    round(grid_search.best_score_, 4)
)


best_pipeline = grid_search.best_estimator_


# ==========================================================
# FINAL TEST
# ==========================================================

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


y_pred = best_pipeline.predict(
    X_test
)

y_score = best_pipeline.decision_function(
    X_test
)


print("\n========================================")
print("SVM FINAL TEST")
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
            y_score
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
    best_pipeline[
        "pca"
    ].n_components_
)