import os, random
import numpy as np
import torch

SEED = 1234  # pick any int and keep it fixed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Make PyTorch as deterministic as possible
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for CUDA determinism on some setups


# notears_mlp_ham10k.py
import numpy as np
import pandas as pd
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ---------- Utilities ----------

def standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sigma, mu, sigma

def acyclicity_constraint(A: torch.Tensor) -> torch.Tensor:
    d = A.shape[0]
    expm = torch.matrix_exp(A * A)
    return torch.trace(expm) - d

def mask_diagonal(M: torch.Tensor) -> torch.Tensor:
    return M * (1.0 - torch.eye(M.shape[0], device=M.device))


# ---------- Models ----------

class MLPNode(nn.Module):
    def __init__(self, d: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


class NOTEARS_MLP(nn.Module):
    def __init__(self, d: int, hidden: int = 16, nonneg_edges: bool = True):
        super().__init__()
        self.d = d
        self.nonneg_edges = nonneg_edges
        self.raw_A = nn.Parameter(torch.zeros(d, d))
        self.f_nodes = nn.ModuleList([MLPNode(d, hidden=hidden) for _ in range(d)])

    def adj_matrix(self) -> torch.Tensor:
        if self.nonneg_edges:
            A = torch.nn.functional.softplus(self.raw_A)
        else:
            A = self.raw_A
        A = mask_diagonal(A)
        return A

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        A = self.adj_matrix()
        N, d = X.shape
        preds = []
        for j in range(d):
            a_j = A[:, j].view(1, -1)
            gated = X * a_j
            yj = self.f_nodes[j](gated)
            preds.append(yj)
        X_hat = torch.cat(preds, dim=1)
        return X_hat, A


# ---------- Training ----------

def train_notears_mlp(
    X: np.ndarray,
    hidden: int = 16,
    batch_size: int = 512,
    lr: float = 1e-3,
    l1_lambda: float = 1e-3,
    max_epochs: int = 2000,
    rho_init: float = 1.0,
    rho_mult: float = 10.0,
    alpha_init: float = 0.0,
    h_tol: float = 1e-8,
    max_outer: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
):
    X_t = torch.from_numpy(X).float().to(device)

    g = torch.Generator()
    g.manual_seed(SEED)
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g, drop_last=False)

    d = X.shape[1]
    model = NOTEARS_MLP(d, hidden=hidden).to(device)

    alpha = alpha_init
    rho = rho_init

    for outer in range(max_outer):
        if verbose:
            print(f"\n[Outer {outer+1}/{max_outer}] rho={rho:.3e}, alpha={alpha:.3e}")

        opt = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, max_epochs + 1):
            model.train()
            total_loss = 0.0
            for (batch_x,) in loader:
                opt.zero_grad()
                xhat, A = model(batch_x)
                mse = ((xhat - batch_x) ** 2).mean()
                l1 = torch.abs(A).sum()

                h = acyclicity_constraint(A)
                loss = mse + l1_lambda * l1 + alpha * h + 0.5 * rho * (h ** 2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total_loss += loss.item() * batch_x.size(0)

            with torch.no_grad():
                _, A_now = model(X_t)
                h_val = acyclicity_constraint(A_now).item()

            if verbose and epoch % 200 == 0:
                print(f"  Epoch {epoch:4d} | loss={total_loss/len(dataset):.6f} | h(A)={h_val:.3e}")

            if abs(h_val) <= h_tol:
                if verbose:
                    print(f"  Early stop inner: h(A)≈0 at epoch {epoch}")
                break

        with torch.no_grad():
            _, A_full = model(X_t)
            h_val = acyclicity_constraint(A_full).item()

        if verbose:
            print(f"End inner loop: h(A)={h_val:.3e}")

        if abs(h_val) <= h_tol:
            if verbose:
                print("Satisfied acyclicity tolerance.")
            break
        else:
            alpha = alpha + rho * h_val
            rho = rho * rho_mult

    with torch.no_grad():
        _, A_final = model(X_t)
        A_final = A_final.cpu().numpy()

    return A_final, model


# ---------- Main with hardcoded args ----------

if __name__ == "__main__":
    # === Hardcoded settings ===
    csv_path = "C:/Users/csj5/Projects/Causal-proxy/Proxy_variables_dir/blastocysts_causal_tags_train_all.csv"   # <-- change this to your CSV file
    hidden_units = 16
    batch_size = 512
    learning_rate = 1e-3
    l1_penalty = 1e-3
    epochs = 2000
    h_tolerance = 1e-8
    rho_initial = 1.0
    rho_multiplier = 10.0
    edge_threshold = 0.05
    verbose = True

    # Load and standardize data
    df = pd.read_csv(csv_path)

    drop_cols = ['Image']
    df = pd.read_csv(csv_path).drop(columns=drop_cols)

    X = df.values.astype(np.float32)
    X, mu, sigma = standardize(X)

    # Train
    A_final, model = train_notears_mlp(
        X,
        hidden=hidden_units,
        batch_size=batch_size,
        lr=learning_rate,
        l1_lambda=l1_penalty,
        max_epochs=epochs,
        h_tol=h_tolerance,
        rho_init=rho_initial,
        rho_mult=rho_multiplier,
        verbose=verbose,
    )

    # Save results
    np.save("Kvasir_notears_mlp_raw.npy", A_final)
    cols = list(df.columns)
    print("\n=== Raw adjacency (A) saved to A_noBlasto_mlp_raw.npy ===")
    print("Top edges (|A_ij| >= threshold):\n")

    edges = []
    d = A_final.shape[0]
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            w = A_final[i, j]
            if abs(w) >= edge_threshold:
                edges.append((cols[i], "->", cols[j], float(w)))

    edges.sort(key=lambda t: abs(t[-1]), reverse=True)
    for u, arrow, v, w in edges[:200]:
        print(f"{u} {arrow} {v}  (w={w:.4f})")

    A_df = pd.DataFrame(A_final, index=cols, columns=cols)
    A_df.to_csv("Blasto_notears_mlp_raw.csv")
    print("\nAdjacency matrix Blasto CSV saved to A_notears_mlp_raw.csv")