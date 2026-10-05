from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from ml_utils import get_raw_data


# ==========================================================
# DATA
# ==========================================================

X_train, X_test, y_train, y_test = get_raw_data()


# ==========================================================
# PIPELINE
# Scaling + PCA happen inside CV folds
# ==========================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    (
        "model",
        SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            max_iter=5000,
            tol=1e-4,
            random_state=42
        )
    )
])


# ==========================================================
# ELASTIC-NET HYPERPARAMETERS
# ==========================================================

param_grid = {
    # Overall regularization strength
    "model__alpha": [
        0.0001,
        0.001,
        0.01,
        0.1
    ],

    # Ratio between L1 and L2
    "model__l1_ratio": [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0
    ]
}


# ==========================================================
# 5-FOLD CV
# ==========================================================

print("\n========================================")
print("ELASTIC-NET LOGISTIC REGRESSION")
print("LEAKAGE-FREE 5-FOLD CV")
print("========================================")


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,

    # Stanford reports F1, accuracy, precision and recall.
    # F1 is used here to choose the final configuration.
    scoring="f1",

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
print("BEST ELASTIC-NET LR PARAMETERS")
print("========================================")

print(grid_search.best_params_)

print(
    "Best CV F1:",
    round(grid_search.best_score_, 4)
)


# ==========================================================
# FINAL TEST
# ==========================================================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(
    X_test
)


print("\n========================================")
print("ELASTIC-NET LR FINAL TEST")
print("========================================")

print(
    "Accuracy:",
    round(
        accuracy_score(y_test, y_pred),
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

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print(
    "\nPCA components:",
    best_model["pca"].n_components_
)