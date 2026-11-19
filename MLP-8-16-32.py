import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

N_RUNS = 30

HIDDEN_SIZES = [
    8,
    16,
    32
]

# ==========================================================
# DATA
# ==========================================================

data = load_breast_cancer()

X = data.data
y = data.target

# ==========================================================
# RUN BENCHMARK
# ==========================================================

for hidden_neurons in HIDDEN_SIZES:

    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    auc_list = []

    for seed in range(N_RUNS):

        # ======================================
        # TRAIN / TEST
        # ======================================

        X_train_full, X_test, y_train_full, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                stratify=y,
                random_state=seed
            )
        )

        # ======================================
        # TRAIN / VALIDATION
        # ======================================

        X_train, X_val, y_train, y_val = (
            train_test_split(
                X_train_full,
                y_train_full,
                test_size=0.20,
                stratify=y_train_full,
                random_state=seed
            )
        )

        # ======================================
        # STANDARDIZATION
        # ======================================

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_val = scaler.transform(
            X_val
        )

        X_test = scaler.transform(
            X_test
        )

        # ======================================
        # FIXED MLP
        # ======================================

        model = MLPClassifier(
            hidden_layer_sizes=(
                hidden_neurons,
            ),
            activation="tanh",
            solver="adam",
            max_iter=1000,
            random_state=seed
        )

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_test
        )

        prob = model.predict_proba(
            X_test
        )[:, 1]

        # ======================================
        # STORE METRICS
        # ======================================

        acc_list.append(
            accuracy_score(
                y_test,
                pred
            )
        )

        prec_list.append(
            precision_score(
                y_test,
                pred,
                average="weighted"
            )
        )

        rec_list.append(
            recall_score(
                y_test,
                pred,
                average="weighted"
            )
        )

        f1_list.append(
            f1_score(
                y_test,
                pred,
                average="weighted"
            )
        )

        auc_list.append(
            roc_auc_score(
                y_test,
                prob
            )
        )

    # ==========================================
    # RESULTS
    # ==========================================

    print("\n")
    print("=" * 70)
    print(
        f"MLP ({hidden_neurons} HIDDEN NEURONS)"
    )
    print("=" * 70)

    print(
        f"Accuracy Mean  : "
        f"{np.mean(acc_list):.4f}"
    )

    print(
        f"Accuracy Std   : "
        f"{np.std(acc_list):.4f}"
    )

    print(
        f"Precision Mean : "
        f"{np.mean(prec_list):.4f}"
    )

    print(
        f"Recall Mean    : "
        f"{np.mean(rec_list):.4f}"
    )

    print(
        f"F1 Mean        : "
        f"{np.mean(f1_list):.4f}"
    )

    print(
        f"F1 Std         : "
        f"{np.std(f1_list):.4f}"
    )

    print(
        f"ROC-AUC Mean   : "
        f"{np.mean(auc_list):.4f}"
    )

    print(
        f"ROC-AUC Std    : "
        f"{np.std(auc_list):.4f}"
    )

    print("=" * 70)

# ==========================================================
# SD-GFNN RESULT FOR COMPARISON
# ==========================================================

print("\n")
print("=" * 70)
print("SD-GFNN")
print("=" * 70)

print("Accuracy Mean            : 0.9766")
print("Accuracy Std             : 0.0151")
print("Precision Mean           : 0.9771")
print("Recall Mean              : 0.9766")
print("F1 Mean                  : 0.9765")
print("F1 Std                   : 0.0151")
print("ROC-AUC Mean             : 0.9946")
print("ROC-AUC Std              : 0.0053")
print("Average Growth Events    : 1.13")
print("Average Best Neurons     : 8.23")
print("Average Terminal Neurons : 9.13")

print("=" * 70)
