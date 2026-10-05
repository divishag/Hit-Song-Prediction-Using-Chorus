from sklearn.ensemble import RandomForestClassifier

from ml_utils import (
    get_data,
    evaluate_model
)


# ==========================================================
# LOAD DATA
# ==========================================================

X_train, X_test, y_train, y_test, pca = get_data()


# ==========================================================
# RANDOM FOREST WITHOUT CLASS WEIGHT
# ==========================================================

rf_normal = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    max_features="log2",
    min_samples_split=2,
    random_state=42
)

evaluate_model(
    "Random Forest Unbalanced",
    rf_normal,
    X_train,
    X_test,
    y_train,
    y_test,
    pca
)


# ==========================================================
# RANDOM FOREST WITH CLASS WEIGHT
# ==========================================================

rf_balanced = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    max_features="log2",
    min_samples_split=2,
    class_weight="balanced",
    random_state=42
)

evaluate_model(
    "Random Forest Balanced",
    rf_balanced,
    X_train,
    X_test,
    y_train,
    y_test,
    pca
)