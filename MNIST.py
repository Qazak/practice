import os
import gzip
import time
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


# ==========================================================
# 1. FAST VECTORIZED FULL-BATCH GRADIENT DESCENT (FLOAT32)
#    Exact implementation of: W_{r}^{t+1} = W_r^t - alpha * grad J_r(W_r^t)
# ==========================================================
class ExactGDNetwork:
    def __init__(self, n_in, n_hidden, n_out, lr=0.08, seed=0):
        rng = np.random.RandomState(seed)
        self.lr = np.float32(lr)
        limit1 = np.sqrt(6.0 / (n_in + n_hidden))
        limit2 = np.sqrt(6.0 / (n_hidden + n_out))
        self.W1 = rng.uniform(-limit1, limit1, size=(n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = rng.uniform(-limit2, limit2, size=(n_hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)

    def _softmax(self, Z):
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shift)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ self.W2 + self.b2
        return Z1, A1, Z2, self._softmax(Z2)

    def fit_gd(self, X, Y_onehot, epochs):
        inv_M = np.float32(1.0 / X.shape[0])
        for _ in range(epochs):
            _, A1, _, P = self.forward(X)
            dZ2 = (P - Y_onehot) * inv_M
            dW2 = A1.T @ dZ2
            db2 = np.sum(dZ2, axis=0)
            dZ1 = (dZ2 @ self.W2.T) * (np.float32(1.0) - A1 * A1)
            self.W1 -= self.lr * (X.T @ dZ1)
            self.b1 -= self.lr * np.sum(dZ1, axis=0)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict(self, X):
        _, _, _, P = self.forward(X)
        return np.argmax(P, axis=1)

    def count_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def synaptic_divergence(self, X):
        _, A1, _, _ = self.forward(X)
        D_j = np.mean(np.abs(A1 - np.mean(A1, axis=0, keepdims=True)), axis=0)
        return D_j, float(np.max(D_j))

    def expand_function_preserving(self, K_new, spot_idx, seed):
        rng = np.random.RandomState(seed)
        u_new = (self.W1[:, [spot_idx]] + rng.normal(0.0, 0.15, size=(self.W1.shape[0], K_new))).astype(np.float32)
        b_new = (self.b1[[spot_idx]] + rng.normal(0.0, 0.15, size=(K_new,))).astype(np.float32)
        v_new = np.zeros((K_new, self.W2.shape[1]), dtype=np.float32)
        self.W1 = np.hstack([self.W1, u_new])
        self.b1 = np.concatenate([self.b1, b_new])
        self.W2 = np.vstack([self.W2, v_new])


def train_gfnn_gd(X_tr, Y_tr_oh, n_classes, mode="full", H0=8, lr=0.08,
                  tau=0.80, max_cycles=5, epochs_per_stage=350, seed=0):
    t0 = time.perf_counter()
    net = ExactGDNetwork(X_tr.shape[1], H0, n_classes, lr=lr, seed=seed)
    net.fit_gd(X_tr, Y_tr_oh, epochs=epochs_per_stage)
    D_j, D_curr = net.synaptic_divergence(X_tr)
    D_init = D_curr
    growth_events = 0

    if mode != "fixed":
        F_prev, F_curr = 1, 1
        for r in range(1, max_cycles + 1):
            should_grow = (D_curr > tau) if mode in ("gfnn_a", "full") else (r <= 2)
            if not should_grow:
                break
            K_r = 1 if mode == "gfnn_a" else F_curr
            spot_idx = int(np.argmax(D_j)) if mode in ("gfnn_a", "full") else (r % len(D_j))
            net.expand_function_preserving(K_r, spot_idx, seed + r * 100)
            growth_events += 1
            net.fit_gd(X_tr, Y_tr_oh, epochs=epochs_per_stage)
            D_j, D_curr = net.synaptic_divergence(X_tr)
            if mode in ("gfnn_b", "full"):
                F_prev, F_curr = F_curr, F_prev + F_curr

    train_ms = (time.perf_counter() - t0) * 1000.0
    return net, train_ms, growth_events, D_init, D_curr


def measure_inference_us(net, X_te, n_repeats=30):
    x_single = X_te[:1]
    net.predict(x_single)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        net.predict(x_single)
    return ((time.perf_counter() - t0) / n_repeats) * 1e6


# ==========================================================
# 2. LOCAL FASHION-MNIST LOADER
# ==========================================================
SEARCH_DIRS = [
    ".",
    r"C:\Users\drdiaz\OneDrive - Prairie View A&M University\Pepers\GFNN 2027",
]


def _read_idx_images(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32)


def _read_idx_labels(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(int)


def load_fashion_mnist(search_dirs, subset_size=1000):
    pairs = [
        ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
        ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
        ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    ]

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        found = False
        for img_name, lbl_name in pairs:
            img_p = os.path.join(d, img_name)
            lbl_p = os.path.join(d, lbl_name)
            if os.path.exists(img_p) and os.path.exists(lbl_p):
                X_all = _read_idx_images(img_p)
                y_all = _read_idx_labels(lbl_p)
                print(f"Loaded Fashion-MNIST from {img_p}", flush=True)
                found = True
                break
        if found:
            break
    else:
        raise FileNotFoundError("Fashion-MNIST IDX pair not found.")

    if subset_size is not None and subset_size < len(X_all):
        X_sub, _, y_sub, _ = train_test_split(
            X_all, y_all, train_size=subset_size, stratify=y_all, random_state=42
        )
        return X_sub, y_sub
    return X_all, y_all


# ==========================================================
# 3. MAIN EXECUTION (~20-30 SECONDS TOTAL IN BROWSER)
# ==========================================================
SUBSET_SIZE = 1000
N_RUNS = 30

X, y = load_fashion_mnist(SEARCH_DIRS, subset_size=SUBSET_SIZE)
n_classes = len(np.unique(y))
print(f"Dataset ready: M={X.shape[0]}, d={X.shape[1]}, C={n_classes}", flush=True)

methods = ["MLP-8", "MLP-16", "MLP-32", "MLP-64", "GFNN-A", "GFNN-B", "Proposed GFNN"]
res = {m: {"acc": [], "f1": [], "params": [], "neurons": [],
           "train_ms": [], "infer_us": [], "growth": [], "d_drop": []} for m in methods}

for seed in range(N_RUNS):
    if seed % 5 == 0:
        print(f"Fashion-MNIST progress: seed {seed + 1}/{N_RUNS}...", flush=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    Y_tr_oh = np.eye(n_classes, dtype=np.float32)[y_tr]

    for h in [8, 16, 32, 64]:
        name = f"MLP-{h}"
        t0 = time.perf_counter()
        net = ExactGDNetwork(X_tr.shape[1], h, n_classes, lr=0.08, seed=seed)
        net.fit_gd(X_tr, Y_tr_oh, epochs=2000)
        t_ms = (time.perf_counter() - t0) * 1000.0
        y_pred = net.predict(X_te)

        res[name]["acc"].append(accuracy_score(y_te, y_pred) * 100.0)
        res[name]["f1"].append(f1_score(y_te, y_pred, average="macro"))
        res[name]["params"].append(net.count_params())
        res[name]["neurons"].append(h)
        res[name]["train_ms"].append(t_ms)
        res[name]["infer_us"].append(measure_inference_us(net, X_te))
        res[name]["growth"].append(0)
        res[name]["d_drop"].append(0.0)

    for mode, name in [("gfnn_a", "GFNN-A"), ("gfnn_b", "GFNN-B"), ("full", "Proposed GFNN")]:
        net, t_ms, g_cnt, d0, dr = train_gfnn_gd(
            X_tr, Y_tr_oh, n_classes, mode=mode, lr=0.08, seed=seed
        )
        y_pred = net.predict(X_te)
        drop_pct = ((d0 - dr) / d0 * 100.0) if d0 > 0 else 0.0

        res[name]["acc"].append(accuracy_score(y_te, y_pred) * 100.0)
        res[name]["f1"].append(f1_score(y_te, y_pred, average="macro"))
        res[name]["params"].append(net.count_params())
        res[name]["neurons"].append(net.W1.shape[1])
        res[name]["train_ms"].append(t_ms)
        res[name]["infer_us"].append(measure_inference_us(net, X_te))
        res[name]["growth"].append(g_cnt)
        res[name]["d_drop"].append(drop_pct)

print("\n" + "=" * 125)
print("DATASET: Fashion-MNIST (Exact Full-Batch Gradient Descent)")
print("=" * 125)
print(f"{'Model':14s} | {'Acc (% ± SD)':14s} | {'F1':6s} | {'Neurons':7s} | {'Params':7s} | "
      f"{'Train(ms)':9s} | {'Infer(us)':9s} | {'Growth':6s} | {'D_drop%':7s} | {'p-val (Wilc / t)'}")
print("-" * 125)

gfnn_acc = np.array(res["Proposed GFNN"]["acc"])
for m in methods:
    acc_arr = np.array(res[m]["acc"])
    if m == "Proposed GFNN":
        p_str = "Reference"
    else:
        diff = gfnn_acc - acc_arr
        if np.allclose(diff, 0):
            p_w, p_t = 1.0, 1.0
        else:
            _, p_w = wilcoxon(gfnn_acc, acc_arr)
            _, p_t = ttest_rel(gfnn_acc, acc_arr)
        p_str = f"{p_w:.4f} / {p_t:.4f}"

    print(f"{m:14s} | {np.mean(acc_arr):5.2f} ± {np.std(acc_arr):4.2f} | "
          f"{np.mean(res[m]['f1']):.4f} | {np.mean(res[m]['neurons']):7.2f} | "
          f"{np.mean(res[m]['params']):7.1f} | {np.mean(res[m]['train_ms']):9.2f} | "
          f"{np.mean(res[m]['infer_us']):9.2f} | {np.mean(res[m]['growth']):6.2f} | "
          f"{np.mean(res[m]['d_drop']):6.1f}% | {p_str}")

print("\nFINISHED")