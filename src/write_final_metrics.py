import pandas as pd
from pathlib import Path

OUT = Path("results/metrics/model_metrics.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "model": "Elastic-Net Logistic Regression",
        "accuracy": 0.7518,
        "precision": 0.7677,
        "recall": 0.8736,
        "f1": 0.8172
    },

    {
        "model": "Shrinkage LDA",
        "accuracy": 0.7518,
        "precision": 0.8272,
        "recall": 0.7701,
        "f1": 0.7976
    },
    {
    "model": "Standard LDA",
    "accuracy": 0.7737,
    "precision": 0.8256,
    "recall": 0.8161,
    "f1": 0.8208
    },

    {
        "model": "SVM",
        "accuracy": 0.7591,
        "precision": 0.8068,
        "recall": 0.8161,
        "f1": 0.8114
    },

    {
        "model": "Random Forest",
        "accuracy": 0.7518,
        "precision": 0.8354,
        "recall": 0.7586,
        "f1": 0.7952
    },

    {
        "model": "Gradient Boosting",
        "accuracy": 0.7153,
        "precision": 0.7449,
        "recall": 0.8391,
        "f1": 0.7892
    },

    {
        "model": "Keras Neural Network",
        "accuracy": 0.7080,
        "precision": 0.7527,
        "recall": 0.8046,
        "f1": 0.7778
    }
]

df = pd.DataFrame(rows)

df.to_csv(
    OUT,
    index=False
)

print(df.to_string(index=False))
print("\nSaved to:", OUT)