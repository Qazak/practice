import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# CONFIGURATION
# ============================================================

N_RUNS = 30
TEST_SIZE = 0.20

# ============================================================
# LOAD DATASET
# ============================================================

data = load_breast_cancer()

X = data.data
y = data.target

# ============================================================
# MULTIPLE RUNS
# ============================================================

acc_list = []
prec_list = []
rec_list = []
f1_list = []

print("=" * 60)
print("RANDOM FOREST BASELINE - BREAST CANCER")
print("=" * 60)

for run in range(N_RUNS):

    random_state = run

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    rec = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    acc_list.append(acc)
    prec_list.append(prec)
    rec_list.append(rec)
    f1_list.append(f1)

    print(
        f"Run {run+1:02d} | "
        f"Accuracy={acc:.4f} | "
        f"F1={f1:.4f}"
    )

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Runs = {N_RUNS}")

print(f"Accuracy Mean  = {np.mean(acc_list):.4f}")
print(f"Accuracy Std   = {np.std(acc_list):.4f}")

print(f"Precision Mean = {np.mean(prec_list):.4f}")
print(f"Precision Std  = {np.std(prec_list):.4f}")

print(f"Recall Mean    = {np.mean(rec_list):.4f}")
print(f"Recall Std     = {np.std(rec_list):.4f}")

print(f"F1 Mean        = {np.mean(f1_list):.4f}")
print(f"F1 Std         = {np.std(f1_list):.4f}")

# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ],
    "Mean": [
        np.mean(acc_list),
        np.mean(prec_list),
        np.mean(rec_list),
        np.mean(f1_list)
    ],
    "Std": [
        np.std(acc_list),
        np.std(prec_list),
        np.std(rec_list),
        np.std(f1_list)
    ]
})

print()
print(results_df)

# ============================================================
# BEST / WORST RUN
# ============================================================

print()
print("=" * 60)
print("ADDITIONAL STATISTICS")
print("=" * 60)

print(f"Best Accuracy  = {np.max(acc_list):.4f}")
print(f"Worst Accuracy = {np.min(acc_list):.4f}")
print(f"Median Accuracy= {np.median(acc_list):.4f}")