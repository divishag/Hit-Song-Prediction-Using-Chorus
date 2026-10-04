from sklearn.linear_model import LogisticRegression

from ml_utils import (
    get_data,
    evaluate_model
)


X_train, X_test, y_train, y_test, pca = get_data()


normal_lr = LogisticRegression(
    max_iter=5000,
    random_state=42
)

evaluate_model(
    "Logistic Regression",
    normal_lr,
    X_train,
    X_test,
    y_train,
    y_test,
    pca
)


balanced_lr = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=42
)

evaluate_model(
    "Logistic Regression Balanced",
    balanced_lr,
    X_train,
    X_test,
    y_train,
    y_test,
    pca
)