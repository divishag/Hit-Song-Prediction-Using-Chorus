from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ml_utils import (
    get_data,
    evaluate_model
)


X_train, X_test, y_train, y_test, pca = get_data()


lda = LinearDiscriminantAnalysis()

evaluate_model(
    "Linear Discriminant Analysis",
    lda,
    X_train,
    X_test,
    y_train,
    y_test,
    pca
)