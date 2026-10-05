from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
# ==========================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    (
        "model",
        LinearDiscriminantAnalysis(
            solver="lsqr"
        )
    )
])


# ==========================================================
# SHRINKAGE VALUES
# ==========================================================

param_grid = {
    "model__shrinkage": [
        None,
        "auto",
        0.1,
        0.25,
        0.5,
        0.75,
        0.9
    ]
}


# ==========================================================
# 5-FOLD CV
# ==========================================================

print("\n========================================")
print("SHRINKAGE LDA")
print("LEAKAGE-FREE 5-FOLD CV")
print("========================================")


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
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
print("BEST SHRINKAGE LDA PARAMETERS")
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
print("SHRINKAGE LDA FINAL TEST")
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