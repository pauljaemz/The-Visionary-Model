"""
================================================================================
VISIONARY — Neural ODE Production Training Script
================================================================================
A continuous-time "physics engine" for the Indian economy.

This script trains a Neural Ordinary Differential Equation (Neural ODE) to learn
the continuous derivatives (laws of motion) of 35 entangled macroeconomic
variables. Instead of discrete sequence models (LSTMs, Transformers), we model
the economy as a continuous dynamical system:

    dy/dt = f_θ(t, y)

where y ∈ R^35 is the economic state vector and f_θ is a learned neural network.

Author : Visionary Project
Target : Kaggle Notebook (GPU recommended)
Dataset: Visionary_Production_Dataset.csv (170 months, 35 variables)
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# ==============================================================================

import os
import time
import warnings
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib

# torchdiffeq — the backbone of our Neural ODE
# Install: pip install torchdiffeq
from torchdiffeq import odeint_adjoint as odeint

warnings.filterwarnings("ignore")

# ---------- Reproducibility ----------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ---------- Device ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[VISIONARY] Using device: {DEVICE}")

# ---------- Paths ----------
# Kaggle input path for the dataset, output goes to /kaggle/working.
# For local runs, override these paths
DATA_PATH = "Visionary_Production_Dataset.csv"
OUTPUT_DIR = "/kaggle/working"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "visionary_ode_model.pth")
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, "visionary_scaler.pkl")

# ---------- Hyperparameters ----------
CONFIG = {
    # Data
    "seq_len": 36,            # Trajectory window: 36 months (3 years)
    "state_dim": 35,          # Number of economic state variables
    "batch_size": 32,         # Increased to 32 to saturate GPU CUDA cores

    # Architecture
    "hidden_dim": 256,        # Width of each hidden layer in ODEFunc MLP
    "num_hidden_layers": 4,   # Depth of the derivative network

    # Training
    "epochs": 300,            # Total training epochs
    "lr": 2e-3,               # Scaled up to compensate for larger batch size
    "weight_decay": 1e-4,     # L2 regularisation strength
    "grad_clip_norm": 1.0,    # Max gradient norm (prevents ODE-induced explosions)

    # ODE Solver
    "solver": "dopri5",       # Runge-Kutta 4(5) adaptive solver
    "rtol": 1e-5,             # Relative tolerance
    "atol": 1e-7,             # Absolute tolerance
}

# ---------- Columns requiring Log10 transform ----------
# These variables exhibit exponential growth and must be log-scaled
# to prevent the neural network from being dominated by their magnitude.
LOG_COLUMNS = [
    "Real_GDP_Crore",
    "Gross_Bank_Credit",
    "Non_Food_Credit",
    "Credit_to_Industry",
    "Credit_to_Services",
    "Personal_Loans",
    "GST_Collection",
    "Nifty50_Price",
    "Forex_Reserves_INR_Crore",
]

print(f"[VISIONARY] Configuration: {CONFIG}")


# ==============================================================================
# SECTION 2: DATASET & DATALOADER
# ==============================================================================

class EconomyTrajectoryDataset(Dataset):
    """
    A PyTorch Dataset that:
      1. Loads the prepared CSV of 170 monthly economic snapshots.
      2. Applies log10 to exponential-growth columns.
      3. Computes FIRST DIFFERENCES (month-over-month deltas) → 169 rows.
      4. Fits a MinMaxScaler on the DELTAS (not absolutes) → bounded training.
      5. Creates sliding-window trajectory samples of consecutive deltas.

    WHY FIRST DIFFERENCES?
      The Neural ODE learns the rate-of-change (Δ) per month, not absolute
      values. GDP typically grows at ~+0.005/month in log space. This is a
      small, stationary signal that the ODE can easily learn and extrapolate.
      At inference, we reconstruct absolute values by cumulative-summing
      the predicted deltas from a known initial state.
    """

    def __init__(self, csv_path: str, seq_len: int = 36, scaler: MinMaxScaler = None):
        """
        Args:
            csv_path:  Path to the prepared CSV file.
            seq_len:   Number of months in each training trajectory.
            scaler:    Pre-fitted MinMaxScaler (if None, a new one is fitted).
        """
        super().__init__()
        self.seq_len = seq_len

        # ------ Load raw data ------
        df = pd.read_csv(csv_path)
        print(f"[DATASET] Loaded {len(df)} rows × {len(df.columns)} columns")

        # Drop the Date column — we only need numeric features
        if "Date" in df.columns:
            df = df.drop(columns=["Date"])

        self.feature_names = list(df.columns)  # Save for later reference
        assert len(self.feature_names) == CONFIG["state_dim"], (
            f"Expected {CONFIG['state_dim']} features, got {len(self.feature_names)}"
        )

        # ------ Step 1: Log10 Transform on exponential-growth variables ------
        # The data is guaranteed to have no exact zeros (replaced with 1e-6),
        # so log10 is safe.
        self.log_columns = [c for c in LOG_COLUMNS if c in df.columns]
        for col in self.log_columns:
            # [FIX] Clip to a tiny positive number to protect against spline overshoots
            df[col] = np.clip(df[col], a_min=1e-6, a_max=None)
            df[col] = np.log10(df[col])
            print(f"  [LOG10] {col}: range [{df[col].min():.4f}, {df[col].max():.4f}]")

        # ------ Step 2: Save the ABSOLUTE log-scaled state for reconstruction ------
        # The last row (July 2025) is the starting point for all future forecasts.
        # We also save the full absolute series for reference.
        data_abs_log = df.values.astype(np.float32)  # (170, 35) absolute log-scaled
        self.last_absolute_state_log = data_abs_log[-1].copy()  # (35,)
        print(f"[DATASET] Saved last absolute state (row {len(data_abs_log)-1}) for reconstruction")

        # ------ Step 3: Compute First Differences ------
        # delta[t] = value[t] - value[t-1] for all 35 variables
        # First row becomes NaN → drop it → 169 rows of deltas
        data_diff = np.diff(data_abs_log, axis=0)  # (169, 35)
        print(f"[DATASET] First differences: {data_diff.shape[0]} rows")
        print(f"  [DIFF] Mean delta range: [{data_diff.mean(axis=0).min():.6f}, "
              f"{data_diff.mean(axis=0).max():.6f}]")

        # ------ Step 3.5: 5-Sigma Winsorization (The GST Shock Fix) ------
        # The GST introduction in July 2017 creates a ~10.33 spike in log-space
        # for GST_Collection. This single outlier distorts the entire MinMaxScaler
        # range, compressing all other deltas to near-zero in scaled space.
        # By clipping to ±5σ before scaling, we preserve 99.99997% of the natural
        # distribution while taming the structural break.
        diff_mean = np.mean(data_diff, axis=0)
        diff_std  = np.std(data_diff, axis=0)

        lower_bound = diff_mean - (5 * diff_std)
        upper_bound = diff_mean + (5 * diff_std)

        n_clipped = np.sum((data_diff < lower_bound) | (data_diff > upper_bound))
        data_diff = np.clip(data_diff, lower_bound, upper_bound)
        print(f"  [WINSORIZE] Clipped {n_clipped} values beyond ±5σ")
        print(f"  [WINSORIZE] Post-clip delta range: [{data_diff.min():.6f}, "
              f"{data_diff.max():.6f}]")
        # ------------------------------------------------------------------

        # ------ Step 4: MinMax Scaling on DELTAS to [0, 1] ------
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.fit(data_diff)
            print("[DATASET] Fitted new MinMaxScaler on DELTAS")
        else:
            self.scaler = scaler
            print("[DATASET] Using pre-fitted MinMaxScaler")

        data_diff_scaled = self.scaler.transform(data_diff)  # (169, 35)

        # ------ Convert to tensor and pre-load to VRAM ------
        self.data_tensor = torch.tensor(data_diff_scaled, dtype=torch.float32).to(DEVICE)

        # ------ Create absolute time span and pre-load to VRAM ------
        # t = [0, 1, 2, ..., seq_len] so dt is ALWAYS exactly 1.0 (1 month)
        self.t_span = torch.arange(0, self.seq_len + 1).float().to(DEVICE)

        # ------ Calculate number of valid sliding windows ------
        # 169 diff rows → n_samples = 169 - seq_len
        self.n_samples = len(self.data_tensor) - self.seq_len
        print(f"[DATASET] Created {self.n_samples} trajectory windows of "
              f"length {self.seq_len + 1} (first-difference mode)")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
            t_span:   (seq_len + 1,) — normalised time points
            y0:       (35,) — initial delta at position `idx`
            y_target: (seq_len + 1, 35) — trajectory of consecutive deltas
        """
        # Extract the trajectory window of deltas: [idx, idx + seq_len]
        trajectory = self.data_tensor[idx : idx + self.seq_len + 1]  # (seq_len+1, 35)
        y0 = trajectory[0]  # (35,) — initial delta

        return self.t_span, y0, trajectory

    def get_scaler(self) -> MinMaxScaler:
        """Returns the fitted scaler (fit on deltas) for inverse transforms."""
        return self.scaler

    def get_log_columns(self) -> list:
        """Returns the list of columns that were log-transformed."""
        return self.log_columns

    def get_feature_names(self) -> list:
        """Returns the ordered list of feature names."""
        return self.feature_names

    def get_last_absolute_state_log(self) -> np.ndarray:
        """Returns the absolute log-scaled state of the last month (July 2025)."""
        return self.last_absolute_state_log.copy()


# ==============================================================================
# SECTION 3: ODEFUNC — THE DERIVATIVE NETWORK
# ==============================================================================

class ODEFunc(nn.Module):
    """
    The learned derivative function f_θ(t, y) → dy/dt.

    This is the "physics engine" — it models the instantaneous rate of change
    of all 35 economic variables as a function of the current state y and time t.

    Architecture:
        - Input:  [y; t] ∈ R^36  (state + time concatenated)
        - Hidden: 4 layers of (Linear → LayerNorm → GELU)
        - Output: dy/dt ∈ R^35

    Design choices:
        - GELU activation: smooth, non-zero gradients everywhere (unlike ReLU)
        - LayerNorm: stabilises the learned dynamics across long ODE trajectories
        - Time-awareness: t is concatenated so dynamics can be non-autonomous
        - No residual/skip on the output: we learn the raw derivative directly
    """

    def __init__(self, state_dim: int = 35, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.state_dim = state_dim

        # Build the MLP layers
        layers = []

        # First layer: (state_dim + 1) → hidden_dim  [+1 for time t]
        layers.append(nn.Linear(state_dim + 1, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())

        # Intermediate hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())

        # Output layer: hidden_dim → state_dim (the derivative dy/dt)
        layers.append(nn.Linear(hidden_dim, state_dim))

        self.net = nn.Sequential(*layers)

        # Initialise output layer with small weights for stable initial dynamics
        nn.init.xavier_normal_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ODEFunc] Architecture: ({state_dim}+1) → "
              f"{' → '.join([str(hidden_dim)] * num_layers)} → {state_dim}")
        print(f"[ODEFunc] Total parameters: {n_params:,}")

    def forward(self, t, y):
        """
        Args:
            t: scalar tensor — current time point
            y: (batch, state_dim) — current state of the economy

        Returns:
            dy/dt: (batch, state_dim) — instantaneous rates of change
        """
        # Expand time t to match batch dimension and concatenate with state
        # t arrives from torchdiffeq as a 0-D scalar tensor (e.g. tensor(0.123)).
        # We must reshape to (1, 1) before expanding to (batch, 1).
        batch_size = y.shape[0]
        t_vec = t.reshape(1, 1).expand(batch_size, 1)  # (batch, 1)

        # Concatenate [y, t] → (batch, state_dim + 1)
        yt = torch.cat([y, t_vec], dim=-1)

        # Forward through the MLP to get dy/dt
        dydt = self.net(yt)

        return dydt


# ==============================================================================
# SECTION 4: NEURAL ODE WRAPPER
# ==============================================================================

class NeuralODE(nn.Module):
    """
    Wraps ODEFunc with the torchdiffeq odeint_adjoint solver.

    Given an initial state y0 and a time span, this module integrates the
    learned dynamics to produce a full predicted trajectory.

    Uses the adjoint method for backpropagation, which is O(1) in memory
    regardless of the number of solver steps — critical for Kaggle's
    memory constraints.
    """

    def __init__(self, func: ODEFunc, solver: str = "dopri5",
                 rtol: float = 1e-5, atol: float = 1e-7):
        super().__init__()
        self.func = func
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(self, y0, t_span):
        """
        Args:
            y0:     (batch, state_dim) — initial state
            t_span: (T,) — time points to solve at

        Returns:
            y_pred: (T, batch, state_dim) — predicted trajectory
        """
        y_pred = odeint(
            self.func,
            y0,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )

        return y_pred


# ==============================================================================
# SECTION 5: TRAINING LOOP
# ==============================================================================

def train():
    """
    Main training function.

    Trains the Neural ODE on 100% of the available data (170 months).
    No train/val/test split — this is the final production model intended
    to learn the complete dynamics for future forecasting.
    """
    print("\n" + "=" * 80)
    print("VISIONARY — Neural ODE Production Training")
    print("=" * 80)

    # ------ 5.1: Create Dataset & DataLoader ------
    dataset = EconomyTrajectoryDataset(
        csv_path=DATA_PATH,
        seq_len=CONFIG["seq_len"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,             # Shuffle trajectories each epoch
        drop_last=False,          # Keep all samples
        num_workers=0,            # Kaggle-safe: no multiprocessing
        pin_memory=False,         # Data is already on the GPU
    )

    print(f"\n[TRAINING] DataLoader: {len(dataloader)} batches/epoch "
          f"(batch_size={CONFIG['batch_size']})")

    # ==========================================================
    # STEADY-STATE ANCHOR SETUP (For Long-Horizon Regularization)
    # ==========================================================
    feature_names = dataset.get_feature_names()
    scaler = dataset.get_scaler()

    # 1. Define the ideal long-term raw deltas (Month-over-Month changes)
    raw_anchor = np.zeros(CONFIG["state_dim"], dtype=np.float32)

    # GDP natural growth: ~6.5% annual = +0.0023 in log10 space
    if "Real_GDP_Crore" in feature_names:
        raw_anchor[feature_names.index("Real_GDP_Crore")] = 0.0023

    # Rates (Inflation, Repo, Deficit, Unemployment) should eventually flatline (delta = 0.0)

    # 2. Scale the raw anchor targets to match the ODE's output space [0, 1]
    scaled_anchor_np = scaler.transform(raw_anchor.reshape(1, -1)).squeeze(0)
    scaled_anchor = torch.tensor(scaled_anchor_np, dtype=torch.float32).to(DEVICE)

    # 3. Create a Mask so we only penalize the 5 core macro variables
    mask_np = np.zeros(CONFIG["state_dim"], dtype=np.float32)
    core_vars = [
        "Real_GDP_Crore", "Inflation_Rate_Monthly_RBI",
        "Repo_Rate", "Gross_Fiscal_Deficit_Percent_GDP",
        "Urban_Youth_Unemployment_Rate",
    ]
    for var in core_vars:
        if var in feature_names:
            mask_np[feature_names.index(var)] = 1.0
    anchor_mask = torch.tensor(mask_np, dtype=torch.float32).to(DEVICE)

    # 4. Create the long-horizon time span (120 months)
    t_span_long = torch.arange(0, 121).float().to(DEVICE)

    # ------ 5.2: Initialise Model ------
    ode_func = ODEFunc(
        state_dim=CONFIG["state_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_hidden_layers"],
    )

    model = NeuralODE(
        func=ode_func,
        solver=CONFIG["solver"],
        rtol=CONFIG["rtol"],
        atol=CONFIG["atol"],
    ).to(DEVICE)

    # ------ 5.3: Optimizer & Scheduler ------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["epochs"],
        eta_min=1e-6,
    )

    # ------ 5.4: Loss Function ------
    criterion = nn.MSELoss()

    # Save the scaler and metadata BEFORE training (bulletproof against interrupts)
    scaler_data = {
        "scaler": dataset.get_scaler(),
        "log_columns": dataset.get_log_columns(),
        "feature_names": dataset.get_feature_names(),
        "config": CONFIG,
        "diff_mode": True,
        "last_absolute_state_log": dataset.get_last_absolute_state_log(),
    }
    joblib.dump(scaler_data, SCALER_SAVE_PATH)
    print(f"[SAVED] Scaler + metadata (diff_mode) → {SCALER_SAVE_PATH}")

    # ------ 5.5: Training Loop ------
    print(f"\n[TRAINING] Starting training for {CONFIG['epochs']} epochs...")
    print(f"[TRAINING] Optimizer: AdamW (lr={CONFIG['lr']}, wd={CONFIG['weight_decay']})")
    print(f"[TRAINING] Scheduler: CosineAnnealingLR (T_max={CONFIG['epochs']})")
    print(f"[TRAINING] Loss: MSE | Grad clip: {CONFIG['grad_clip_norm']}")
    print("-" * 80)

    best_loss = float("inf")
    train_start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (t_span, y0, y_target) in enumerate(dataloader):
            # Data is already pre-loaded on DEVICE via the Dataset class

            # Forward pass: integrate the ODE from y0 over 120 MONTHS
            y_pred_long = model(y0, t_span_long)           # (121, batch, 35)
            y_pred_long = y_pred_long.permute(1, 0, 2)     # (batch, 121, 35)

            # Split into Historical (0-36) and Future (37-120)
            y_pred_historical = y_pred_long[:, :CONFIG["seq_len"] + 1, :]
            y_pred_future     = y_pred_long[:, CONFIG["seq_len"] + 1:, :]

            # Loss 1: Standard historical trajectory matching
            loss_historical = criterion(y_pred_historical, y_target)

            # Loss 2: Long-Horizon Anchor Penalty (Mean Reversion)
            future_diff = (y_pred_future - scaled_anchor) * anchor_mask
            loss_anchor = torch.mean(future_diff ** 2)

            # Total Loss (lambda = 0.1 to act as a gentle regularizer)
            loss = loss_historical + (0.1 * loss_anchor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — essential for ODE stability
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=CONFIG["grad_clip_norm"],
            )

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Step the learning rate scheduler
        scheduler.step()

        # Compute epoch metrics
        avg_loss = epoch_loss / n_batches
        current_lr = scheduler.get_last_lr()[0]

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # Log progress every 10 epochs (or first/last epoch)
        if epoch == 1 or epoch % 10 == 0 or epoch == CONFIG["epochs"]:
            elapsed = time.time() - train_start
            print(
                f"  Epoch [{epoch:>4d}/{CONFIG['epochs']}] | "
                f"Loss: {avg_loss:.8f} | "
                f"Best: {best_loss:.8f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

    # ------ 5.6: Final Save ------
    train_elapsed = time.time() - train_start
    print("-" * 80)
    print(f"[TRAINING] Complete! Total time: {train_elapsed:.1f}s")
    print(f"[TRAINING] Best loss: {best_loss:.8f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[SAVED] Model weights → {MODEL_SAVE_PATH}")

    # ------ 5.7: Sanity Check — Reconstruct one trajectory ------
    print("\n[SANITY CHECK] Reconstructing first trajectory from trained model...")
    model.eval()
    with torch.no_grad():
        t_span, y0, y_target = dataset[0]
        t_span = t_span.to(DEVICE)
        y0 = y0.unsqueeze(0).to(DEVICE)       # (1, 35)
        y_target = y_target.unsqueeze(0)       # (1, seq_len+1, 35)

        y_pred = model(y0, t_span)             # (seq_len+1, 1, 35)
        y_pred = y_pred.permute(1, 0, 2).cpu() # (1, seq_len+1, 35)

        final_mse = criterion(y_pred, y_target).item()
        print(f"  First trajectory MSE: {final_mse:.8f}")

        # Per-variable reconstruction error at the final time step
        pred_final = y_pred[0, -1, :].numpy()
        true_final = y_target[0, -1, :].numpy()
        max_err_idx = np.argmax(np.abs(pred_final - true_final))
        max_err_var = dataset.get_feature_names()[max_err_idx]
        max_err_val = np.abs(pred_final - true_final)[max_err_idx]
        print(f"  Largest final-step error: {max_err_var} = {max_err_val:.6f} (scaled)")

    print("\n" + "=" * 80)
    print("VISIONARY — Training Complete. Model ready for Agentic simulation.")
    print("=" * 80)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    train()
