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
)

# ==========================================================
# SYNAPTIC DIVERGENCE
# ==========================================================

def compute_synaptic_divergence(model, X):
    hidden_input = np.dot(X, model.coefs_[0]) + model.intercepts_[0]

    hidden_output = np.tanh(hidden_input)

    mean_activation = np.mean(hidden_output, axis=0)

    divergence = np.mean(
        np.abs(hidden_output - mean_activation),
        axis=0
    )

    return divergence


# ==========================================================
# SD-GFNN
# ==========================================================

class SDGFNN:

    def __init__(
        self,
        tau=0.80,
        initial_neurons=8,
        max_growth_rounds=10,
        random_state=0
    ):

        self.tau = tau
        self.initial_neurons = initial_neurons
        self.max_growth_rounds = max_growth_rounds
        self.random_state = random_state

        self.best_model = None
        self.best_val_acc = -1

        self.final_neurons = None
        self.final_dmax = None

        self.growth_events = 0
        self.history = []

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val
    ):

        neurons = self.initial_neurons

        fib_prev = 1
        fib_curr = 1

        for growth_round in range(self.max_growth_rounds):

            model = MLPClassifier(
                hidden_layer_sizes=(neurons,),
                activation="tanh",
                solver="adam",
                max_iter=1000,
                random_state=self.random_state
            )

            model.fit(X_train, y_train)

            # ======================================
            # VALIDATION PERFORMANCE
            # ======================================

            val_pred = model.predict(X_val)

            val_acc = accuracy_score(
                y_val,
                val_pred
            )

            # ======================================
            # SYNAPTIC DIVERGENCE
            # ======================================

            divergence_values = compute_synaptic_divergence(
                model,
                X_val
            )

            d_max = np.max(divergence_values)
            d_mean = np.mean(divergence_values)

            self.history.append({
                "round": growth_round + 1,
                "neurons": neurons,
                "dmax": d_max,
                "dmean": d_mean,
                "val_acc": val_acc
            })

            print(
                f"Round {growth_round + 1:2d} | "
                f"Neurons={neurons:3d} | "
                f"Dmax={d_max:.4f} | "
                f"Dmean={d_mean:.4f} | "
                f"ValAcc={val_acc:.4f}"
            )

            # ======================================
            # SAVE BEST MODEL
            # ======================================

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model = model
                self.final_neurons = neurons
                self.final_dmax = d_max

            # ======================================
            # THEORETICAL STOPPING CONDITION
            # ======================================

            if d_max <= self.tau:
                break

            # ======================================
            # FIBONACCI GROWTH
            # ======================================

            if growth_round == 0:
                neurons += 1

            elif growth_round == 1:
                neurons += 1

            else:
                new_neurons = fib_prev + fib_curr

                fib_prev, fib_curr = (
                    fib_curr,
                    new_neurons
                )

                neurons += new_neurons

            self.growth_events += 1

    def predict(self, X):
        return self.best_model.predict(X)


# ==========================================================
# DATASET
# ==========================================================

data = load_breast_cancer()

X = data.data
y = data.target

# ==========================================================
# STORAGE
# ==========================================================

N_RUNS = 30

acc_list = []
prec_list = []
rec_list = []
f1_list = []

growth_events = []
final_neurons = []

# ==========================================================
# EXPERIMENTS
# ==========================================================

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
    # SCALING
    # ======================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ======================================
    # SD-GFNN
    # ======================================

    model = SDGFNN(
        tau=0.80,
        initial_neurons=8,
        max_growth_rounds=10,
        random_state=seed
    )

    model.fit(
        X_train,
        y_train,
        X_val,
        y_val
    )

    pred = model.predict(X_test)

    acc_list.append(
        accuracy_score(y_test, pred)
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

    growth_events.append(
        model.growth_events
    )

    final_neurons.append(
        model.final_neurons
    )

# ==========================================================
# RESULTS
# ==========================================================

print("\n")
print("=" * 60)
print("SD-GFNN RESULTS")
print("=" * 60)

print(
    f"Accuracy Mean = {np.mean(acc_list):.4f}"
)

print(
    f"Accuracy Std  = {np.std(acc_list):.4f}"
)

print(
    f"Precision Mean = {np.mean(prec_list):.4f}"
)

print(
    f"Recall Mean    = {np.mean(rec_list):.4f}"
)

print(
    f"F1 Mean        = {np.mean(f1_list):.4f}"
)

print(
    f"F1 Std         = {np.std(f1_list):.4f}"
)

print(
    f"Average Growth Events = "
    f"{np.mean(growth_events):.2f}"
)

print(
    f"Average Final Neurons = "
    f"{np.mean(final_neurons):.2f}"
)

print("\nSD-GFNN ACCURACY")
print(
    list(
        np.round(
            acc_list,
            6
        )
    )
)

print("\nSD-GFNN F1")
print(
    list(
        np.round(
            f1_list,
            6
        )
    )
)