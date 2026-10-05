import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
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
# REPRODUCIBILITY
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================================
# LOAD DATA
# ==========================================================

X_train, X_test, y_train, y_test = get_raw_data()


# ==========================================================
# BUILD KERAS MODEL
# Stanford-style:
# 2 hidden layers + dropout + L2 regularization
# ==========================================================

def build_model(
    input_dim,
    hidden_1,
    hidden_2,
    dropout_rate,
    l2_strength
):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(input_dim,)
        ),

        tf.keras.layers.Dense(
            hidden_1,
            activation="relu",
            kernel_regularizer=
                tf.keras.regularizers.l2(
                    l2_strength
                )
        ),

        tf.keras.layers.Dropout(
            dropout_rate
        ),

        tf.keras.layers.Dense(
            hidden_2,
            activation="relu",
            kernel_regularizer=
                tf.keras.regularizers.l2(
                    l2_strength
                )
        ),

        tf.keras.layers.Dropout(
            dropout_rate
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])


    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==========================================================
# PARAMETERS TO TEST
#
# Stanford did not publish exact final values,
# so we tune reasonable candidates.
# ==========================================================

hidden_sizes = [
    (32, 16),
    (64, 32),
    (128, 64)
]

dropout_rates = [
    0.2,
    0.4
]

l2_strengths = [
    0.0001,
    0.001
]


# ==========================================================
# LEAKAGE-FREE 5-FOLD CV
# ==========================================================

print("\n========================================")
print("KERAS NEURAL NETWORK")
print("LEAKAGE-FREE 5-FOLD CV")
print("========================================")


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


best_score = -1
best_params = None


total_configs = (
    len(hidden_sizes)
    * len(dropout_rates)
    * len(l2_strengths)
)

print(
    "Configurations:",
    total_configs
)

print(
    "Total fits:",
    total_configs * 5
)


# ==========================================================
# GRID SEARCH MANUALLY
# ==========================================================

for hidden_1, hidden_2 in hidden_sizes:

    for dropout_rate in dropout_rates:

        for l2_strength in l2_strengths:

            fold_scores = []


            print(
                "\nTesting:",
                f"layers=({hidden_1},{hidden_2}),",
                f"dropout={dropout_rate},",
                f"l2={l2_strength}"
            )


            for fold, (
                train_idx,
                val_idx
            ) in enumerate(
                skf.split(
                    X_train,
                    y_train
                ),
                start=1
            ):

                # ------------------------------------------
                # SPLIT CURRENT CV FOLD
                # ------------------------------------------

                X_fold_train = X_train[
                    train_idx
                ]

                X_fold_val = X_train[
                    val_idx
                ]

                y_fold_train = y_train[
                    train_idx
                ]

                y_fold_val = y_train[
                    val_idx
                ]


                # ------------------------------------------
                # SCALE INSIDE FOLD
                # ------------------------------------------

                scaler = StandardScaler()

                X_fold_train = (
                    scaler.fit_transform(
                        X_fold_train
                    )
                )

                X_fold_val = (
                    scaler.transform(
                        X_fold_val
                    )
                )


                # ------------------------------------------
                # PCA INSIDE FOLD
                # ------------------------------------------

                pca = PCA(
                    n_components=0.95
                )

                X_fold_train = (
                    pca.fit_transform(
                        X_fold_train
                    )
                )

                X_fold_val = (
                    pca.transform(
                        X_fold_val
                    )
                )


                # ------------------------------------------
                # RESET TF STATE
                # ------------------------------------------

                tf.keras.backend.clear_session()

                tf.random.set_seed(SEED)


                # ------------------------------------------
                # BUILD MODEL
                # ------------------------------------------

                model = build_model(
                    input_dim=
                        X_fold_train.shape[1],

                    hidden_1=hidden_1,
                    hidden_2=hidden_2,

                    dropout_rate=
                        dropout_rate,

                    l2_strength=
                        l2_strength
                )


                # ------------------------------------------
                # TRAIN
                # ------------------------------------------

                model.fit(
                    X_fold_train,
                    y_fold_train,

                    epochs=100,
                    batch_size=32,

                    verbose=0
                )


                # ------------------------------------------
                # VALIDATE
                # ------------------------------------------

                probabilities = (
                    model.predict(
                        X_fold_val,
                        verbose=0
                    ).ravel()
                )

                predictions = (
                    probabilities >= 0.5
                ).astype(int)


                score = f1_score(
                    y_fold_val,
                    predictions
                )

                fold_scores.append(
                    score
                )


            mean_score = np.mean(
                fold_scores
            )

            std_score = np.std(
                fold_scores
            )


            print(
                "Mean CV F1:",
                round(mean_score, 4),
                "| std:",
                round(std_score, 4)
            )


            # ----------------------------------------------
            # BEST CONFIGURATION
            # ----------------------------------------------

            if mean_score > best_score:

                best_score = mean_score

                best_params = {
                    "hidden_1":
                        hidden_1,

                    "hidden_2":
                        hidden_2,

                    "dropout_rate":
                        dropout_rate,

                    "l2_strength":
                        l2_strength
                }


# ==========================================================
# BEST CV RESULT
# ==========================================================

print("\n========================================")
print("BEST KERAS NN PARAMETERS")
print("========================================")

print(best_params)

print(
    "Best CV F1:",
    round(best_score, 4)
)


# ==========================================================
# FINAL PREPROCESSING
#
# Fit ONLY using full training set.
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


pca = PCA(
    n_components=0.95
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)


# ==========================================================
# FINAL MODEL
# ==========================================================

tf.keras.backend.clear_session()

tf.random.set_seed(SEED)


final_model = build_model(

    input_dim=
        X_train_pca.shape[1],

    hidden_1=
        best_params["hidden_1"],

    hidden_2=
        best_params["hidden_2"],

    dropout_rate=
        best_params["dropout_rate"],

    l2_strength=
        best_params["l2_strength"]
)


final_model.fit(
    X_train_pca,
    y_train,

    epochs=100,
    batch_size=32,

    verbose=0
)


# ==========================================================
# FINAL TEST
# ==========================================================

probabilities = final_model.predict(
    X_test_pca,
    verbose=0
).ravel()


y_pred = (
    probabilities >= 0.5
).astype(int)


print("\n========================================")
print("KERAS NN FINAL TEST")
print("========================================")


print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            y_pred
        ),
        4
    )
)

print(
    "F1:",
    round(
        f1_score(
            y_test,
            y_pred
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            y_pred
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            y_pred
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
    "\nPCA components:",
    X_train_pca.shape[1]
)