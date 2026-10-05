import pandas as pd
from pathlib import Path


OUTPUT = Path("results/metrics/best_params.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


rows = [
    {
        "model": "SVM",
        "best_parameters": "C=0.01, kernel=linear",
        "cv_metric": "balanced_accuracy",
        "best_cv_score": 0.6647
    },

    {
        "model": "Random Forest",
        "best_parameters": (
            "max_depth=5, "
            "max_features=log2, "
            "min_samples_split=2, "
            "n_estimators=200"
        ),
        "cv_metric": "balanced_accuracy",
        "best_cv_score": 0.6358
    },

    {
        "model": "Gradient Boosting",
        "best_parameters": (
            "learning_rate=0.1, "
            "max_depth=1, "
            "n_estimators=100, "
            "subsample=0.8"
        ),
        "cv_metric": "balanced_accuracy",
        "best_cv_score": 0.6026
    },

    {
        "model": "Neural Network",
        "best_parameters": (
            "activation=relu, "
            "alpha=0.0001, "
            "hidden_layer_sizes=(64, 32), "
            "learning_rate_init=0.01"
        ),
        "cv_metric": "balanced_accuracy",
        "best_cv_score": 0.6502
    }
]


df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT,
    index=False
)

print("\n========================================")
print("BEST PARAMETERS SAVED")
print("========================================")

print(df)

print("\nSaved to:", OUTPUT)