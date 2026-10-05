import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier

from ml_utils import (
    get_data,
    get_test_metadata
)


OUTPUT_DIR = Path("results/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# LOAD SAME TRAIN / TEST DATA
# ==========================================================

X_train, X_test, y_train, y_test, pca = get_data()

metadata = get_test_metadata()


# ==========================================================
# SAVE ONE MODEL'S PREDICTIONS
# ==========================================================

def save_predictions(
    name,
    model,
    filename
):

    print("\n========================================")
    print(name)
    print("========================================")

    model.fit(
        X_train,
        y_train
    )

    predicted = model.predict(
        X_test
    )

    # score/probability used for ranking predictions
    if hasattr(model, "predict_proba"):

        score = model.predict_proba(
            X_test
        )[:, 1]

    elif hasattr(model, "decision_function"):

        score = model.decision_function(
            X_test
        )

    else:

        score = [None] * len(y_test)


    output = metadata.copy()

    output["actual"] = y_test
    output["predicted"] = predicted
    output["score"] = score

    output["correct"] = (
        output["actual"]
        == output["predicted"]
    )


    path = OUTPUT_DIR / filename

    output.to_csv(
        path,
        index=False
    )


    print(
        "Predictions:",
        len(output)
    )

    print(
        "Correct:",
        output["correct"].sum()
    )

    print(
        "Incorrect:",
        (~output["correct"]).sum()
    )

    print(
        "Saved to:",
        path
    )


# ==========================================================
# 1. LOGISTIC REGRESSION
# ==========================================================

logistic = LogisticRegression(
    max_iter=5000,
    random_state=42
)

save_predictions(
    "Logistic Regression",
    logistic,
    "logistic_regression_predictions.csv"
)


# ==========================================================
# 2. BALANCED LOGISTIC REGRESSION
# ==========================================================

logistic_balanced = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=42
)

save_predictions(
    "Logistic Regression Balanced",
    logistic_balanced,
    "logistic_regression_balanced_predictions.csv"
)


# ==========================================================
# 3. LDA
# ==========================================================

lda = LinearDiscriminantAnalysis()

save_predictions(
    "Linear Discriminant Analysis",
    lda,
    "lda_predictions.csv"
)


# ==========================================================
# 4. SVM
#
# Best CV params:
# C = 0.01
# kernel = linear
# ==========================================================

svm = SVC(
    C=0.01,
    kernel="linear",
    random_state=42
)

save_predictions(
    "SVM",
    svm,
    "svm_predictions.csv"
)


# ==========================================================
# 5. RANDOM FOREST
#
# Best CV params
# ==========================================================

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    max_features="log2",
    min_samples_split=2,
    class_weight="balanced",
    random_state=42
)

save_predictions(
    "Random Forest",
    random_forest,
    "random_forest_predictions.csv"
)


# ==========================================================
# 6. GRADIENT BOOSTING
#
# Best CV params
# ==========================================================

gradient_boosting = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=1,
    n_estimators=100,
    subsample=0.8,
    random_state=42
)

save_predictions(
    "Gradient Boosting",
    gradient_boosting,
    "gradient_boosting_predictions.csv"
)


# ==========================================================
# 7. NEURAL NETWORK
#
# Best CV params
# ==========================================================

neural_network = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    alpha=0.0001,
    learning_rate_init=0.01,
    max_iter=2000,
    early_stopping=True,
    random_state=42
)

save_predictions(
    "Neural Network",
    neural_network,
    "neural_network_predictions.csv"
)


print("\n========================================")
print("ALL PREDICTIONS SAVED")
print("========================================")