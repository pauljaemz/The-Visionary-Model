"""
================================================================================
VISIONARY — Multi-Agent Orchestration Framework
================================================================================
A 4-Agent "Optimal Control" policy dashboard for the Indian economy.

Pipeline:
  1. StrategistAgent  — Parses natural language goal → PolicyObjective (Pydantic)
  2. ArchitectAgent   — Neural ODE physics engine with policy injection
  3. OptimizerAgent   — Optuna-based policy search over the ODE simulator
  4. AnalystAgent     — Translates optimal policy → executive markdown report

Usage:
    orchestrator = VisionaryOrchestrator(
        model_path="visionary_ode_model.pth",
        scaler_path="visionary_scaler.pkl",
        data_path="Final_Visionary_Economy_Dataset_Prepared.csv",
        openai_api_key="sk-..."  # or None for mock mode
    )
    report = orchestrator.run("Achieve $10T GDP by 2035 with inflation under 6%")
    print(report)

Author : Visionary Project
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import json
import logging
import time
import warnings
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import optuna

# Pydantic for strict structured output
from pydantic import BaseModel, Field

# OpenAI SDK — mock-safe (imported conditionally)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# torchdiffeq for inference (regular odeint, not adjoint — no backprop needed)
from torchdiffeq import odeint

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VISIONARY")

# ==============================================================================
# DEVICE
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Columns that were log10-transformed during training
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

# The default policy levers that Optuna can adjust
DEFAULT_LEVERS = [
    "Repo_Rate",
    "Gross_Fiscal_Deficit_Percent_GDP",
]

# GDP conversion: 1 USD Trillion ≈ 83 * 1e5 Crore (at ~₹83/USD)
# More precisely: 1 Trillion USD = 1e12 USD * 83 INR/USD / 1e7 (crore) = 83e5 Crore
USD_TO_CRORE_FACTOR = 83.0e5  # 83 lakh crore per trillion USD

# Converts 2011 Base Year Real GDP to Current Nominal GDP (approx 1.65x today)
# plus estimated future inflation compounding (~4% per year).
# For a 2028-2030 horizon, a multiplier of 2.15 is highly accurate.
REAL_TO_NOMINAL_MULTIPLIER = 2.15 


# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================

class PolicyObjective(BaseModel):
    """
    Strict mathematical objective extracted from a natural language policy goal.
    Every field has a sensible default so the system never crashes on partial input.
    """
    target_year: int = Field(
        default=2035,
        description="The year by which the GDP target should be reached."
    )
    target_gdp_trillions: float = Field(
        default=10.0,
        description="Target GDP in trillions of USD."
    )
    max_inflation_pct: float = Field(
        default=6.0,
        description="Maximum allowable monthly inflation rate (%)."
    )
    max_unemployment_pct: float = Field(
        default=25.0,
        description="Maximum allowable urban youth unemployment rate (%)."
    )
    allowed_levers: List[str] = Field(
        default_factory=lambda: DEFAULT_LEVERS.copy(),
        description="List of economic variable names that can be adjusted as policy."
    )


# ==============================================================================
# MODEL ARCHITECTURE (must match training script exactly)
# ==============================================================================

class ODEFunc(nn.Module):
    """
    The learned derivative function f_θ(t, y) → dy/dt.
    Architecture must be identical to the training script for weight loading.
    """

    def __init__(self, state_dim: int = 35, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.state_dim = state_dim
        layers = []
        layers.append(nn.Linear(state_dim + 1, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        batch_size = y.shape[0]
        # [FIX] Cap time at 36.0 to prevent out-of-distribution hallucination
        # during long-term forecasts (e.g., 2033 = 96 months).
        t_safe = torch.clamp(t, max=36.0)
        t_vec = t_safe.reshape(1, 1).expand(batch_size, 1)
        yt = torch.cat([y, t_vec], dim=-1)
        return self.net(yt)


class NeuralODE(nn.Module):
    """Wraps ODEFunc with torchdiffeq odeint for inference."""

    def __init__(self, func: ODEFunc, solver: str = "dopri5",
                 rtol: float = 1e-5, atol: float = 1e-7):
        super().__init__()
        self.func = func
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(self, y0, t_span):
        return odeint(
            self.func, y0, t_span,
            method=self.solver, rtol=self.rtol, atol=self.atol,
        )


# ==============================================================================
# AGENT 3: THE ARCHITECT — Physics Engine
# ==============================================================================

class ArchitectAgent:
    """
    The core simulation engine. Loads the trained Neural ODE and provides
    a simulate_trajectory() method that supports piecewise policy injection.

    =========================================================================
    DUAL-STATE ACCUMULATOR ARCHITECTURE
    =========================================================================

    The Neural ODE was trained on SCALED FIRST-DIFFERENCES (month-over-month
    deltas of log-transformed data, then MinMaxScaled to [0,1]).

    At inference, TWO parallel state tracks are maintained:

    TRACK 1 — "Scaled Delta Space" (PyTorch / ODE domain)
      - This is what the ODE sees and predicts.
      - Values live in [0, 1] (MinMaxScaled deltas).
      - The ODE input at each step is a scaled delta vector.
      - The ODE output at each step is ALSO a scaled delta vector.

    TRACK 2 — "Absolute Log Space" (Python / NumPy accumulator)
      - A running Python array that holds the TRUE absolute state
        of the economy in log-space (for log columns) / raw-space
        (for non-log columns).
      - Updated each month via: abs[t] = abs[t-1] + inverse_scale(delta[t])
      - This is NEVER fed into the ODE.

    POLICY INTERVENTION MATH:
      When Optuna requests Repo_Rate = 6.5% at month 12:
        1. Read current_absolute_state["Repo_Rate"] = e.g. 7.0
        2. Compute required_delta = 6.5 - 7.0 = -0.5
        3. Scale this delta using the MinMaxScaler fit on deltas
        4. Inject this SCALED DELTA into the ODE state tensor
        5. Set current_absolute_state["Repo_Rate"] = 6.5
      The ODE NEVER sees the number 6.5. It only sees a scaled delta.

    FINAL OUTPUT:
      After all months are accumulated, apply 10^x to the log_columns
      to get real-world Crore values. Non-log columns pass through as-is.
    =========================================================================
    """

    def __init__(self, model_path: str, scaler_path: str, data_path: str):
        """
        Args:
            model_path:  Path to visionary_ode_model.pth
            scaler_path: Path to visionary_scaler.pkl
            data_path:   Path to the original CSV (for date + seed delta)
        """
        logger.info("ArchitectAgent: Initialising physics engine...")

        # ------ Load scaler metadata ------
        try:
            scaler_data = joblib.load(scaler_path)
            self.scaler: MinMaxScaler = scaler_data["scaler"]  # Fit on DELTAS
            self.log_columns: List[str] = scaler_data["log_columns"]
            self.feature_names: List[str] = scaler_data["feature_names"]
            self.config: dict = scaler_data["config"]
            self.diff_mode: bool = scaler_data.get("diff_mode", True)

            # TRACK 2 ANCHOR: The absolute log-scaled state at July 2025.
            # This is the cumulative-sum starting point for ALL forecasts.
            # For log_columns this is log10(value), for others it's the raw value.
            self.initial_abs_log: np.ndarray = scaler_data[
                "last_absolute_state_log"
            ].copy()

            logger.info(f"  Scaler loaded: {len(self.feature_names)} features "
                        f"(diff_mode={self.diff_mode})")
        except Exception as e:
            logger.error(f"  Failed to load scaler: {e}")
            raise

        # Feature name → index mapping for fast lookup
        self.feature_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.feature_names)
        }
        self.state_dim = len(self.feature_names)

        # ------ Reconstruct and load model ------
        try:
            ode_func = ODEFunc(
                state_dim=self.config.get("state_dim", 35),
                hidden_dim=self.config.get("hidden_dim", 256),
                num_layers=self.config.get("num_hidden_layers", 4),
            )
            self.model = NeuralODE(
                func=ode_func,
                solver=self.config.get("solver", "dopri5"),
                rtol=self.config.get("rtol", 1e-5),
                atol=self.config.get("atol", 1e-7),
            ).to(DEVICE)

            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            n_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"  Model loaded: {n_params:,} parameters")
        except Exception as e:
            logger.error(f"  Failed to load model: {e}")
            raise

        # ------ Extract last date from the CSV ------
        try:
            df_dates = pd.read_csv(data_path)
            if "Date" in df_dates.columns:
                self.last_date = pd.to_datetime(df_dates["Date"].iloc[-1])
            else:
                self.last_date = pd.to_datetime("2025-07-01")
            logger.info(f"  Base date: {self.last_date.strftime('%B %Y')}")
        except Exception as e:
            logger.warning(f"  Could not read date from CSV: {e}")
            self.last_date = pd.to_datetime("2025-07-01")

        # ------ Compute TRACK 1 seed: last historical scaled delta ------
        # The ODE needs a scaled delta as its y0. We compute:
        #   raw_delta = abs_log[July 2025] - abs_log[June 2025]
        #   scaled_delta = scaler.transform(raw_delta)
        # This is the "momentum" the economy had going into the forecast.
        try:
            df_raw = pd.read_csv(data_path)
            if "Date" in df_raw.columns:
                df_raw = df_raw.drop(columns=["Date"])
            for col in self.log_columns:
                if col in df_raw.columns:
                    df_raw[col] = np.log10(df_raw[col])
            data_abs = df_raw.values.astype(np.float32)
            # raw delta = July 2025 - June 2025 (in log-space)
            last_delta_raw = data_abs[-1] - data_abs[-2]  # (36,)
            self.initial_delta_scaled = self.scaler.transform(
                last_delta_raw.reshape(1, -1)
            ).squeeze(0).astype(np.float32)  # (36,)
            logger.info("  Seed delta computed: July 2025 - June 2025")
        except Exception as e:
            logger.warning(f"  Could not compute seed delta: {e}. Using scaler midpoint.")
            self.initial_delta_scaled = np.full(
                self.state_dim, 0.5, dtype=np.float32
            )

        logger.info("ArchitectAgent: Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_initial_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the two initial states needed for simulation:

        Returns:
            scaled_delta_y0:     (36,) — last historical scaled delta.
                                 This is the ODE's initial condition (Track 1).
            absolute_base_state: (36,) — absolute log-scaled state at July 2025.
                                 This is the cumsum anchor (Track 2).
        """
        return (
            self.initial_delta_scaled.copy(),
            self.initial_abs_log.copy(),
        )

    def assimilate_partial_state(
        self, target_month: str, user_overrides: Dict[str, Dict[str, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        State Initialization via Partial Observability.
        
        Args:
            target_month: Target initialization month (e.g., "2026-03").
            user_overrides: Dict in format { "Feature_Name": { "t": value, "t_minus_1": value } }
            
        Returns:
            Tuple of (new_scaled_delta, absolute_base_state) to pass into simulate_trajectory
            or use as the starting point for Optuna.
        """
        target_date = pd.to_datetime(target_month)
        
        # 1. Calculate months to simulate from base date (July 2025)
        months_to_simulate = (target_date.year - self.last_date.year) * 12 + (target_date.month - self.last_date.month)
        
        if months_to_simulate < 1:
            logger.warning(f"Target month {target_month} is on or before base date {self.last_date.strftime('%Y-%m')}. Falling back to baseline initial state.")
            return self.get_initial_state()

        logger.info(f"Assimilating partial state for target origin {target_month}...")
        
        # 2. Forward-Simulation Imputation Engine
        # Generate baseline trajectory from our historical endpoint up to target_month
        df_imputed = self.simulate_trajectory(self.get_initial_state(), months=months_to_simulate)
        
        # 3. State-Merge Logic (Data Assimilation)
        # df_imputed contains absolute real-world values
        # Index -1 corresponds to target_month (t)
        # Index -2 corresponds to target_month - 1 month (t-1)
        merged_t_real = df_imputed.iloc[-1].copy()
        merged_t_minus_1_real = df_imputed.iloc[-2].copy()
        
        # Overwrite with user-provided real-world ground truth (if available)
        count_t = 0
        count_t_minus_1 = 0
        for feat, values in user_overrides.items():
            if feat in merged_t_real.index:
                t_val = values.get("t")
                tm1_val = values.get("t_minus_1")
                
                if t_val is not None and pd.notna(t_val):
                    merged_t_real[feat] = t_val
                    count_t += 1
                if tm1_val is not None and pd.notna(tm1_val):
                    merged_t_minus_1_real[feat] = tm1_val
                    count_t_minus_1 += 1
                
        # 4. Matrix Re-Initialization
        # Convert merged records back into the "Track 2 API" accumulator representation
        # i.e., applying log10 to volumetric columns like GDP and Credit
        merged_t_abs_log = merged_t_real.values.astype(np.float64)
        merged_t_minus_1_abs_log = merged_t_minus_1_real.values.astype(np.float64)
        
        for col in self.log_columns:
            idx = self.feature_idx[col]
            merged_t_abs_log[idx] = np.log10(max(merged_t_real.iloc[idx], 1e-6))
            merged_t_minus_1_abs_log[idx] = np.log10(max(merged_t_minus_1_real.iloc[idx], 1e-6))
            
        # Compute first-difference mathematical velocity
        raw_delta = merged_t_abs_log - merged_t_minus_1_abs_log
        
        # Pass through the min-max scaler fit on the training data
        new_scaled_delta = self._scale_delta(raw_delta)
        
        # [VELOCITY SHIELD] — Prevent numerical explosion from extreme user inputs
        new_scaled_delta = np.clip(new_scaled_delta, 0.0, 1.0)
        
        logger.info(f"  Assimilated {count_t} inputs for t and "
                    f"{count_t_minus_1} for t-1.")
        
        # Returns exactly what standard `.get_initial_state()` returns, but shifted to target_month
        return (new_scaled_delta, merged_t_abs_log)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inverse_scale_delta(self, scaled_delta: np.ndarray) -> np.ndarray:
        """
        Inverse MinMaxScale a single scaled delta vector back to raw delta
        (log-space for log columns, raw-space for non-log columns).

        Args:
            scaled_delta: (36,) array in [0,1] scaled-delta space
        Returns:
            raw_delta: (36,) array in real delta units (log-space)
        """
        return self.scaler.inverse_transform(
            scaled_delta.reshape(1, -1)
        ).squeeze(0)

    def _scale_delta(self, raw_delta: np.ndarray) -> np.ndarray:
        """
        MinMaxScale a raw delta vector into scaled-delta space [0,1].

        Args:
            raw_delta: (36,) array in real delta units (log-space)
        Returns:
            scaled_delta: (36,) array in [0,1] scaled-delta space
        """
        return self.scaler.transform(
            raw_delta.reshape(1, -1)
        ).squeeze(0).astype(np.float32)

    def _abs_log_to_real(self, abs_log_trajectory: np.ndarray) -> np.ndarray:
        """
        FINAL STEP: Convert accumulated absolute log-space trajectory to
        real-world values. Applies 10^x ONLY to log_columns; all other
        columns pass through unchanged.

        Args:
            abs_log_trajectory: (T, 36) — accumulated absolute values
                                (log-space for log cols, raw for others)
        Returns:
            trajectory_real:    (T, 36) — real-world units
        """
        trajectory_real = abs_log_trajectory.copy()
        for col in self.log_columns:
            idx = self.feature_idx[col]
            trajectory_real[:, idx] = np.power(10, trajectory_real[:, idx])
        return trajectory_real

    def get_driving_forces(self, initial_state: tuple) -> dict:
        """
        Extract the underlying mathematical derivatives (velocities) from the ODE
        for the very next month.

        Args:
            initial_state: Tuple of (scaled_delta_y0, absolute_base_state)
        Returns:
            dict containing the Top 5 positive forces and Top 5 negative forces.
        """
        # 1. Unpack initial state
        if isinstance(initial_state, tuple) and len(initial_state) == 2:
            scaled_delta_y0, absolute_base = initial_state
        else:
            scaled_delta_y0 = np.asarray(initial_state, dtype=np.float32)

        # 2. PyTorch ode_state & t_span for 1 step
        ode_state = torch.tensor(
            scaled_delta_y0, dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)
        
        t_span = torch.arange(0, 2).float().to(DEVICE)

        # 3. Predict exactly 1 step forward
        with torch.no_grad():
            y_pred = self.model(ode_state, t_span)
        
        # y_pred[1] is the predicted scaled delta
        pred_scaled_delta = y_pred[1].squeeze(0).cpu().numpy()
        
        # Clip to replicate velocity shield
        pred_scaled_delta = np.clip(pred_scaled_delta, 0.0, 1.0)

        # 4. Inverse-scale to get raw_delta (mathematical velocity)
        raw_delta = self._inverse_scale_delta(pred_scaled_delta)

        # 5. Map to feature names
        forces = {
            feat: float(raw_delta[idx]) 
            for feat, idx in self.feature_idx.items()
        }

        # 6. Sort and get Top 5 positive and Top 5 negative forces
        sorted_forces = sorted(forces.items(), key=lambda item: item[1])
        
        top_negative = dict(sorted_forces[:5])
        top_positive = dict(sorted_forces[-5:])
        
        # Reverse top positive to show highest first
        top_positive = {k: v for k, v in reversed(top_positive.items())}

        return {
            "top_positive_forces": top_positive,
            "top_negative_forces": top_negative
        }

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def simulate_trajectory(
        self,
        initial_state: Any,
        months: int,
        policy_interventions: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """
        Simulate the economy forward for `months` months using the Neural ODE.

        =====================================================================
        DUAL-STATE ACCUMULATOR LOOP
        =====================================================================

        STEP-BY-STEP for each month in a segment:

            1. The ODE predicts: y_pred[i] = a SCALED DELTA  (Track 1)
            2. Inverse-scale:    raw_delta = scaler.inverse_transform(y_pred[i])
            3. Accumulate:       current_abs += raw_delta     (Track 2)
            4. Store current_abs in the trajectory list.

        At an intervention month:
            1. Read current_abs[lever_idx]  (e.g., Repo_Rate = 7.0)
            2. Compute required_raw_delta = target_value - current_abs[lever_idx]
               For log columns: target must be converted to log10 first.
            3. Scale the required delta: scaled = scaler.transform(raw_delta)
            4. Inject scaled delta into ODE state tensor (Track 1)
            5. Set current_abs[lever_idx] = target_value     (Track 2)
            6. Resume odeint

        =====================================================================

        Args:
            initial_state:        Tuple of (scaled_delta_y0, absolute_base_state)
                                  OR (36,) scaled delta (for backward compat —
                                  absolute_base_state defaults to self.initial_abs_log).
            months:               Number of months to simulate forward.
            policy_interventions: Dict mapping month offsets to dicts of
                                  {feature_name: real_world_value}.
                                  e.g. {12: {"Repo_Rate": 6.5}}

        Returns:
            pd.DataFrame with (months+1) rows, 36 columns of REAL-WORLD values,
            indexed by monthly dates starting from the base date.
        """
        if policy_interventions is None:
            policy_interventions = {}

        # ------ Unpack initial states ------
        # Support both tuple (new API) and bare array (backward compat)
        if isinstance(initial_state, tuple) and len(initial_state) == 2:
            scaled_delta_y0, absolute_base = initial_state
        else:
            # Backward compat: assume it's just the scaled delta
            scaled_delta_y0 = np.asarray(initial_state, dtype=np.float32)
            absolute_base = self.initial_abs_log.copy()

        # =====================================================================
        # TRACK 1: ODE state — lives in scaled-delta space [0, 1]
        # This tensor is what odeint reads and writes.
        # =====================================================================
        ode_state = torch.tensor(
            scaled_delta_y0, dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)  # (1, 36)

        # =====================================================================
        # TRACK 2: Absolute accumulator — lives in log-space / raw-space
        # A plain NumPy array. NEVER fed into the ODE.
        # Updated each month via: current_abs += inverse_scale(predicted_delta)
        # =====================================================================
        current_abs = absolute_base.copy().astype(np.float64)  # (36,)

        # Trajectory collector: list of (36,) absolute log-space snapshots
        # Month 0 = the base state (July 2025)
        trajectory = [current_abs.copy()]

        # ------ Build piecewise integration segments ------
        breakpoints = sorted(set(
            [0] + [m for m in policy_interventions.keys() if 0 < m < months] + [months]
        ))

        with torch.no_grad():
            for seg_idx in range(len(breakpoints) - 1):
                seg_start = breakpoints[seg_idx]
                seg_end = breakpoints[seg_idx + 1]
                seg_months = seg_end - seg_start

                if seg_months <= 0:
                    continue

                # ==========================================================
                # INTERVENTION INJECTION (if this breakpoint has one)
                # ==========================================================
                if seg_start in policy_interventions and seg_start > 0:
                    interventions = policy_interventions[seg_start]

                    # We'll build a raw delta vector that represents ALL the
                    # lever changes at this intervention point. Non-intervened
                    # features keep the ODE's last predicted delta.
                    #
                    # Start from the ODE's current state (its last predicted
                    # scaled delta) and inverse-scale it to get the "natural"
                    # raw delta the ODE wanted to apply.
                    ode_state_np = ode_state.cpu().numpy().squeeze(0)  # (36,)
                    raw_delta_for_step = self._inverse_scale_delta(ode_state_np)

                    for feat_name, target_real_val in interventions.items():
                        if feat_name not in self.feature_idx:
                            logger.warning(
                                f"  Unknown feature '{feat_name}' — skipping"
                            )
                            continue

                        idx = self.feature_idx[feat_name]

                        # Convert target to the same space as current_abs
                        # (log10 for log columns, raw for others)
                        if feat_name in self.log_columns:
                            target_in_log = np.log10(max(target_real_val, 1e-6))
                        else:
                            target_in_log = target_real_val

                        # THE KEY MATH:
                        # required_delta = where_we_want_to_be - where_we_are
                        required_raw_delta = target_in_log - current_abs[idx]
                        raw_delta_for_step[idx] = required_raw_delta

                        # Update Track 2: snap the absolute state to the target
                        current_abs[idx] = target_in_log

                    # Scale the combined raw delta → new ODE state (Track 1)
                    new_scaled_delta = self._scale_delta(raw_delta_for_step)

                    # [VELOCITY SHIELD] — Clip the scaled delta to the valid
                    # ODE input range [0, 1]. Large policy jumps (e.g. Repo
                    # Rate changing by 1.5%) can produce scaled values >> 1.0,
                    # which would shatter the neural network weights and cause
                    # exponential accumulation. The clip keeps the ODE stable.
                    # Track 2 (current_abs) is NOT clipped — the absolute
                    # tracking remains 100% accurate to the requested target.
                    new_scaled_delta = np.clip(new_scaled_delta, 0.0, 1.0)

                    ode_state = torch.tensor(
                        new_scaled_delta, dtype=torch.float32
                    ).unsqueeze(0).to(DEVICE)

                    # DO NOT append to trajectory here — the intervention
                    # modifies the state AT this month, which was already
                    # appended at the end of the previous segment.
                    # We just update the last entry to reflect the snap.
                    trajectory[-1] = current_abs.copy()

                # ==========================================================
                # ODE INTEGRATION for this segment
                # ==========================================================
                # Time span normalised to [0, 1] (matches training)
                # Time span in absolute months (dt = 1.0) to match training
                n_steps = seg_months + 1
                t_span = torch.arange(0, n_steps).float().to(DEVICE)

                try:
                    # y_pred shape: (n_steps, 1, 36) — all in SCALED DELTA space
                    y_pred = self.model(ode_state, t_span)
                    y_pred = y_pred.squeeze(1).cpu().numpy()  # (n_steps, 36)
                except Exception as e:
                    logger.error(
                        f"  ODE integration failed [{seg_start}→{seg_end}]: {e}"
                    )
                    # Graceful fallback: flat extrapolation (zero delta)
                    for _ in range(seg_months):
                        trajectory.append(current_abs.copy())
                    continue

                # ==========================================================
                # DELTA → ABSOLUTE ACCUMULATION (the cumsum)
                # ==========================================================
                # y_pred[0] is the ODE's re-emission of its input (≈ ode_state).
                # y_pred[1..n_steps-1] are the predicted deltas for each month.
                for i in range(1, n_steps):
                    # Step 1: Get the ODE's raw predicted scaled delta
                    predicted_scaled_delta = y_pred[i]  # (36,)

                    # [VELOCITY SHIELD] — Clip ALL ODE outputs to [0, 1].
                    # During long extrapolation (100+ months), the neural
                    # network drifts outside its training range. Unclipped
                    # deltas compound exponentially in the accumulator,
                    # causing quintillion-dollar GDP explosions. This clip
                    # constrains every predicted delta to the range the
                    # scaler was originally fit on.
                    predicted_scaled_delta = np.clip(
                        predicted_scaled_delta, 0.0, 1.0
                    )

                    # Step 2: Inverse-scale to get the real delta (log-space)
                    raw_delta = self._inverse_scale_delta(predicted_scaled_delta)

                    # ==========================================================
                    # [REAL-WORLD MACROECONOMIC HARD-CAPS & STEADY STATES]
                    # ==========================================================
                    
                    # 1. GDP Structural Drift (The Tailwind)
                    # Adds ~6.5% annual real growth (+0.0023 per month in log10 space)
                    if "Real_GDP_Crore" in self.feature_idx:
                        gdp_idx = self.feature_idx["Real_GDP_Crore"]
                        raw_delta[gdp_idx] += 0.0023
                        # Raise max cap to 0.005 so the tailwind and stimulus have room to compound
                        raw_delta[gdp_idx] = np.clip(raw_delta[gdp_idx], -0.005, 0.005)

                    # 2. RBI Inflation Anchor
                    # RBI Statutory Target is 4% (Band: 2% to 6%)
                    if "Inflation_Rate_Monthly_RBI" in self.feature_idx:
                        inf_idx = self.feature_idx["Inflation_Rate_Monthly_RBI"]
                        raw_delta[inf_idx] = np.clip(raw_delta[inf_idx], -0.5, 0.5)
                        if current_abs[inf_idx] + raw_delta[inf_idx] < -2.0:
                            raw_delta[inf_idx] = -2.0 - current_abs[inf_idx]

                    # 3. Repo Rate Floor (Zero Lower Bound)
                    if "Repo_Rate" in self.feature_idx:
                        repo_idx = self.feature_idx["Repo_Rate"]
                        if current_abs[repo_idx] + raw_delta[repo_idx] < 3.0:
                            raw_delta[repo_idx] = 3.0 - current_abs[repo_idx]

                    # 4. Fiscal Deficit Bounds (FRBM Act)
                    if "Gross_Fiscal_Deficit_Percent_GDP" in self.feature_idx:
                        def_idx = self.feature_idx["Gross_Fiscal_Deficit_Percent_GDP"]
                        next_val = current_abs[def_idx] + raw_delta[def_idx]
                        if next_val < 3.0:
                            raw_delta[def_idx] = 3.0 - current_abs[def_idx]
                        elif next_val > 10.0:
                            raw_delta[def_idx] = 10.0 - current_abs[def_idx]

                    # 5. Unemployment Floor
                    if "Urban_Youth_Unemployment_Rate" in self.feature_idx:
                        unemp_idx = self.feature_idx["Urban_Youth_Unemployment_Rate"]
                        raw_delta[unemp_idx] = np.clip(raw_delta[unemp_idx], -0.5, 0.5)
                        if current_abs[unemp_idx] + raw_delta[unemp_idx] < 4.0:
                            raw_delta[unemp_idx] = 4.0 - current_abs[unemp_idx]

                    # Step 3: Accumulate into Track 2
                    # abs[t] = abs[t-1] + delta[t]
                    current_abs = current_abs + raw_delta

                    # ==========================================================
                    # 6. STOCK-FLOW CONSISTENCY (ACCOUNTING IDENTITY)
                    # ==========================================================
                    # MUST HAPPEN AFTER ACCUMULATION TO OVERWRITE THE ODE'S DELTA
                    gva_cols = [c for c in self.feature_names if 'GVA_' in c]
                    total_gva = sum(current_abs[self.feature_idx[c]] for c in gva_cols if c in self.feature_idx)
                    
                    total_gva = max(total_gva, 1e-6)
                    
                    if "Real_GDP_Crore" in self.feature_idx:
                        gdp_idx = self.feature_idx["Real_GDP_Crore"]
                        true_gdp = total_gva / 0.9123
                        current_abs[gdp_idx] = np.log10(true_gdp)

                    # Step 4: Store the snapshot
                    trajectory.append(current_abs.copy())

                # [VELOCITY SHIELD] — Clip the carryover state before
                # re-seeding the ODE for the next segment.
                last_pred = np.clip(y_pred[-1], 0.0, 1.0)
                ode_state = torch.tensor(
                    last_pred, dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)

        # ==================================================================
        # FINAL: Convert accumulated absolute log-space → real-world values
        # ==================================================================
        # Stack: (months+1, 36) in absolute log-space
        trajectory_abs_log = np.stack(trajectory, axis=0)

        # Apply 10^x ONLY to log_columns (GDP, Credit, etc.)
        # Non-log columns (Repo_Rate, Inflation, etc.) pass through as-is.
        trajectory_real = self._abs_log_to_real(trajectory_abs_log)

        # Build the output DataFrame with monthly date index
        dates = [
            self.last_date + relativedelta(months=i)
            for i in range(len(trajectory_real))
        ]
        df = pd.DataFrame(
            trajectory_real, columns=self.feature_names, index=dates
        )
        df.index.name = "Date"

        return df


# ==============================================================================
# AGENT 1: THE STRATEGIST — Goal Parser (LLM + Pydantic)
# ==============================================================================

class StrategistAgent:
    """
    Parses a natural language policy goal into a strict mathematical
    PolicyObjective using an LLM (GPT-4o) + Pydantic validation.

    If no API key is provided or the LLM call fails, falls back to
    mock mode with sensible defaults.
    """

    # The system prompt instructs the LLM to output strict JSON
    SYSTEM_PROMPT = """You are an expert macroeconomic policy analyst.
The user will provide a natural language policy goal for the Indian economy.
You must extract a strict mathematical objective from it.

You MUST respond with ONLY a JSON object (no markdown, no explanation) with these fields:
{
    "target_year": <int, the year by which the target should be reached>,
    "target_gdp_trillions": <float, target GDP in trillions of USD>,
    "max_inflation_pct": <float, maximum allowable inflation rate in %>,
    "max_unemployment_pct": <float, maximum allowable urban youth unemployment rate in %>,
    "allowed_levers": <list of strings from the following allowed values:
        "Repo_Rate", "Gross_Fiscal_Deficit_Percent_GDP",
        "Gross_Fixed_Capital_Formation_Percent_GDP", "GSec_10Y_Yield">
}

Rules:
- If the user does not specify a value, use these defaults:
  target_year=2035, target_gdp_trillions=10.0, max_inflation_pct=6.0,
  max_unemployment_pct=25.0, allowed_levers=["Repo_Rate", "Gross_Fiscal_Deficit_Percent_GDP"]
- GDP is always in USD trillions.
- Inflation is the monthly RBI CPI-based rate.
- Respond with ONLY valid JSON. No extra text."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        self.mock_mode = True

        if api_key and OPENAI_AVAILABLE:
            try:
                # Rerouted to Hugging Face Free Serverless Inference
                self.client = OpenAI(
                    base_url="https://router.huggingface.co/v1/",
                    api_key=api_key 
                )
                self.mock_mode = False
                logger.info("StrategistAgent: LLM mode (Llama-3.3-70B-Instruct)")
            except Exception as e:
                logger.warning(f"StrategistAgent: OpenAI init failed ({e}), using mock mode")
        else:
            if not OPENAI_AVAILABLE:
                logger.info("StrategistAgent: openai package not installed — mock mode")
            else:
                logger.info("StrategistAgent: No API key — mock mode")

    def parse_goal(self, user_prompt: str) -> PolicyObjective:
        """
        Parse a natural language goal into a PolicyObjective.

        Args:
            user_prompt: e.g. "Achieve $10T GDP by 2035 with inflation under 6%"

        Returns:
            PolicyObjective with extracted fields
        """
        logger.info(f"StrategistAgent: Parsing goal — \"{user_prompt[:80]}...\"")

        if self.mock_mode:
            return self._mock_parse(user_prompt)

        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )

            raw_json = response.choices[0].message.content.strip()
            # Clean any markdown fencing the LLM might have added
            if raw_json.startswith("```"):
                raw_json = raw_json.split("\n", 1)[1]
                raw_json = raw_json.rsplit("```", 1)[0]

            objective = PolicyObjective.model_validate_json(raw_json)
            logger.info(f"  Parsed: GDP=${objective.target_gdp_trillions}T by "
                        f"{objective.target_year}, inflation<{objective.max_inflation_pct}%")
            return objective

        except Exception as e:
            logger.warning(f"  LLM parse failed ({e}), falling back to mock mode")
            return self._mock_parse(user_prompt)

    def _mock_parse(self, user_prompt: str) -> PolicyObjective:
        """
        Deterministic mock parser that extracts numbers from the prompt
        using simple heuristics, with sensible defaults.
        """
        import re

        objective = PolicyObjective()  # Start with defaults

        # Try to extract target GDP (look for patterns like "$10T", "10 trillion")
        gdp_match = re.search(r'\$?\s*(\d+(?:\.\d+)?)\s*[Tt](?:rillion)?', user_prompt)
        if gdp_match:
            objective.target_gdp_trillions = float(gdp_match.group(1))

        # Try to extract target year (4-digit year > 2025)
        year_match = re.search(r'\b(202[5-9]|20[3-9]\d)\b', user_prompt)
        if year_match:
            objective.target_year = int(year_match.group(1))

        # Try to extract inflation constraint
        infl_match = re.search(
            r'inflation\s*(?:under|below|<|less than|max)?\s*(\d+(?:\.\d+)?)\s*%',
            user_prompt, re.IGNORECASE
        )
        if infl_match:
            objective.max_inflation_pct = float(infl_match.group(1))

        # Try to extract unemployment constraint
        unemp_match = re.search(
            r'unemployment\s*(?:under|below|<|less than|max)?\s*(\d+(?:\.\d+)?)\s*%',
            user_prompt, re.IGNORECASE
        )
        if unemp_match:
            objective.max_unemployment_pct = float(unemp_match.group(1))

        logger.info(f"  [MOCK] Parsed: GDP=${objective.target_gdp_trillions}T by "
                    f"{objective.target_year}, inflation<{objective.max_inflation_pct}%, "
                    f"unemployment<{objective.max_unemployment_pct}%")
        return objective


# ==============================================================================
# AGENT 2: THE OPTIMIZER — Policy Search (Optuna)
# ==============================================================================

class OptimizerAgent:
    """
    Uses Optuna (Tree-structured Parzen Estimator) to search for the optimal
    sequence of policy interventions that achieves the PolicyObjective while
    respecting the inflation and unemployment constraints.
    """

    def __init__(self, architect: ArchitectAgent, objective: PolicyObjective, initial_state_override: Optional[Tuple] = None):
        self.architect = architect
        self.objective = objective
        self.initial_state_tuple = initial_state_override if initial_state_override is not None else self.architect.get_initial_state()

        # Calculate forecast horizon in months
        # We forecast from July 2025 (or target origin) to December of target_year
        if initial_state_override is not None and hasattr(self, 'target_month') :
             pass # Date calculation should actually just rely on the end goal. 
             # Wait, self.architect.last_date is fixed at July 2025. 
             # For forecast horizon, it's safer to just calculate from architect.last_date as it's the absolute base, 
             # and the simulation handles trajectory starting from the base. But wait, if we assimilate, the simulation 
             # still starts from the base, just the "intervention" window shifts? 
             # No, if we pass initial_state_override, ArchitectAgent.simulate_trajectory expects to simulate forward from that state.
             # Wait, `assimilate_partial_state` returns a state that corresponds to `target_month`. So simulation should be fewer months!
        
        # Actually it's simpler: Let's fix forecast_months to always be from current date if no override, or from target_month if override.
        # Wait, the user said they are implementing this for UI, so we can just let Optuna run from the base date, but Optuna needs to know the number of months.
        # Let's adjust target start date.
        
        current_year = self.architect.last_date.year
        current_month = self.architect.last_date.month
        target_end = datetime(objective.target_year, 12, 1)
        current_start = datetime(current_year, current_month, 1)
        self.forecast_months = (target_end.year - current_start.year) * 12 + \
                               (target_end.month - current_start.month)

        if self.forecast_months <= 0:
            logger.warning("Target year is in the past — adjusting to 120 months (10 years)")
            self.forecast_months = 120

        # Intervention frequency: every 3 months (Quarterly)
        self.intervention_freq = 3
        self.n_interventions = max(1, self.forecast_months // self.intervention_freq)

        # Convert GDP target to Crore for comparison with the model output
        self.target_gdp_crore = objective.target_gdp_trillions * USD_TO_CRORE_FACTOR

        # Build index lookups for constraint variables
        self.gdp_idx = self.architect.feature_idx.get("Real_GDP_Crore", 0)
        self.inflation_idx = self.architect.feature_idx.get("Inflation_Rate_Monthly_RBI", 18)
        self.unemployment_idx = self.architect.feature_idx.get(
            "Urban_Youth_Unemployment_Rate", 24
        )

        # Define search bounds for each lever
        self._lever_bounds = {
            "Repo_Rate": (3.0, 9.0),
            "Gross_Fiscal_Deficit_Percent_GDP": (3.0, 10.0),
            "Gross_Fixed_Capital_Formation_Percent_GDP": (25.0, 40.0),
            "GSec_10Y_Yield": (5.0, 9.0),
        }

        logger.info(f"OptimizerAgent: Forecast horizon = {self.forecast_months} months "
                    f"({self.n_interventions} intervention points)")
        logger.info(f"  Target GDP: {objective.target_gdp_trillions}T USD "
                    f"= {self.target_gdp_crore:,.0f} Crore")
        logger.info(f"  Constraints: inflation < {objective.max_inflation_pct}%, "
                    f"unemployment < {objective.max_unemployment_pct}%")
        logger.info(f"  Levers: {objective.allowed_levers}")

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function. Samples policy interventions, runs the
        simulation and returns a score to MINIMISE.

        Score = -reward_gdp + penalty_inflation + penalty_unemployment

        Lower is better (Optuna minimises by default).
        """
        # ------ Sample policy interventions ------
        policy_interventions: Dict[int, Dict[str, float]] = {}

        for i in range(self.n_interventions):
            month = (i + 1) * self.intervention_freq
            if month > self.forecast_months:
                break

            interventions_at_month = {}
            for lever in self.objective.allowed_levers:
                if lever in self._lever_bounds:
                    low, high = self._lever_bounds[lever]
                    value = trial.suggest_float(
                        f"{lever}_m{month}", low, high
                    )
                    interventions_at_month[lever] = value

            if interventions_at_month:
                policy_interventions[month] = interventions_at_month

        # ------ Run simulation ------
        try:
            trajectory_df = self.architect.simulate_trajectory(
                initial_state=self.initial_state_tuple,
                months=self.forecast_months,
                policy_interventions=policy_interventions,
            )
        except Exception as e:
            logger.debug(f"  Trial {trial.number} simulation failed: {e}")
            return 1e6  # Large penalty for failed simulations

        # ------ Compute score ------

        # 1. GDP Reward: Removed max(0.0) to allow massive NEGATIVE scores (rewards) for growth
        target_year_df = trajectory_df[trajectory_df.index.year == self.objective.target_year]
        if target_year_df.empty:
            target_year_df = trajectory_df.iloc[-12:] 

        annual_gdp = target_year_df["Real_GDP_Crore"].sum()
        gdp_ratio = annual_gdp / self.target_gdp_crore
        
        # Massive incentive to push ratio as high as possible. Optuna loves negative scores.
        gdp_penalty = (1.0 - gdp_ratio) * 2000.0  

        # 2. Inflation Penalty
        inflation_col = "Inflation_Rate_Monthly_RBI"
        if inflation_col in trajectory_df.columns:
            inflation_violations = trajectory_df[inflation_col].apply(
                lambda x: max(0.0, x - self.objective.max_inflation_pct)
            ).sum()
        else:
            inflation_violations = 0.0

        # 3. Unemployment Penalty
        unemployment_col = "Urban_Youth_Unemployment_Rate"
        if unemployment_col in trajectory_df.columns:
            unemployment_violations = trajectory_df[unemployment_col].apply(
                lambda x: max(0.0, x - self.objective.max_unemployment_pct)
            ).sum()
        else:
            unemployment_violations = 0.0

        # 4. Stability Penalty
        stability_penalty = 0.0
        sorted_months = sorted(policy_interventions.keys())
        for lever in self.objective.allowed_levers:
            previous_value = None
            for month in sorted_months:
                if lever in policy_interventions[month]:
                    current_value = policy_interventions[month][lever]
                    if previous_value is not None:
                        delta = abs(current_value - previous_value)
                        if delta > 0.5:  
                            # Reduced from 1000.0 to 50.0 so Optuna is not terrified to move rates
                            stability_penalty += 50.0 * (delta - 0.5)
                    previous_value = current_value

        # Total score (lower is better, negative is best)
        score = (gdp_penalty
                 + 25.0 * inflation_violations  # Strong wall against hyperinflation
                 + 5.0 * unemployment_violations
                 + stability_penalty)

        return score

    def optimize(self, n_trials: int = 500, progress_callback=None) -> Tuple[
        Dict[int, Dict[str, float]], pd.DataFrame, pd.DataFrame
    ]:
        """
        Run the Optuna optimization.

        Args:
            n_trials: Number of Optuna trials to run.

        Returns:
            Tuple of:
                - best_policy: Dict[month, Dict[lever_name, value]]
                - optimal_trajectory: pd.DataFrame of the best policy's trajectory
                - baseline_trajectory: pd.DataFrame of no-intervention trajectory
        """
        logger.info(f"OptimizerAgent: Running baseline simulation (no interventions)...")

        # ------ Baseline trajectory (no interventions) ------
        baseline_trajectory = self.architect.simulate_trajectory(
            initial_state=self.initial_state_tuple,
            months=self.forecast_months,
            policy_interventions=None,
        )
        target_year_df = baseline_trajectory[baseline_trajectory.index.year == self.objective.target_year]
        if target_year_df.empty:
            target_year_df = baseline_trajectory.iloc[-12:]
        baseline_gdp = target_year_df["Real_GDP_Crore"].sum()
        
        logger.info(f"  Baseline target year GDP: {baseline_gdp:,.0f} Crore "
                    f"(${baseline_gdp / USD_TO_CRORE_FACTOR:.2f}T)")

        # ------ Create Optuna study ------
        logger.info(f"OptimizerAgent: Starting Optuna search ({n_trials} trials)...")
        start_time = time.time()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name="visionary_policy_search",
        )

        study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[progress_callback] if progress_callback else None
        )

        elapsed = time.time() - start_time
        logger.info(f"  Search complete in {elapsed:.1f}s")
        logger.info(f"  Best score: {study.best_value:.4f}")

        # ------ Reconstruct best policy ------
        best_params = study.best_params
        best_policy: Dict[int, Dict[str, float]] = {}

        for i in range(self.n_interventions):
            month = (i + 1) * self.intervention_freq
            if month > self.forecast_months:
                break

            interventions_at_month = {}
            for lever in self.objective.allowed_levers:
                key = f"{lever}_m{month}"
                if key in best_params:
                    interventions_at_month[lever] = best_params[key]

            if interventions_at_month:
                best_policy[month] = interventions_at_month

        # ------ Log the best policy ------
        logger.info("  Best policy sequence:")
        for month, interventions in sorted(best_policy.items()):
            date = self.architect.last_date + relativedelta(months=month)
            parts = [f"{k}={v:.2f}" for k, v in interventions.items()]
            logger.info(f"    Month {month} ({date.strftime('%b %Y')}): {', '.join(parts)}")

        # ------ Re-simulate with best policy for the final trajectory ------
        optimal_trajectory = self.architect.simulate_trajectory(
            initial_state=self.initial_state_tuple,
            months=self.forecast_months,
            policy_interventions=best_policy,
        )
        target_year_df = optimal_trajectory[optimal_trajectory.index.year == self.objective.target_year]
        if target_year_df.empty:
            target_year_df = optimal_trajectory.iloc[-12:]
        optimal_gdp = target_year_df["Real_GDP_Crore"].sum()
        
        logger.info(f"  Optimal target year GDP: {optimal_gdp:,.0f} Crore "
                    f"(${optimal_gdp / USD_TO_CRORE_FACTOR:.2f}T)")

        # ------ BASELINE SAFETY CHECK ------
        # If the optimizer's best policy mathematically underperforms doing nothing, discard it.
        if optimal_gdp <= baseline_gdp:
            logger.warning("  Optuna failed to beat the baseline. Reverting to Baseline (No Policy).")
            best_policy = {}
            optimal_trajectory = baseline_trajectory
            optimal_gdp = baseline_gdp

        return best_policy, optimal_trajectory, baseline_trajectory, study


# ==============================================================================
# AGENT 4: THE ANALYST — Report Generator (LLM)
# ==============================================================================

class AnalystAgent:
    """
    Translates the optimal policy sequence and trajectory comparison into a
    professional executive markdown report, using an LLM or a template fallback.
    """

    SYSTEM_PROMPT = """You are a senior macroeconomic policy advisor writing an executive
brief for the Prime Minister's Economic Advisory Council.

You will receive structured data containing:
1. The policy objective (target GDP, constraints)
2. The optimal policy timeline found by the optimization engine
3. Key economic indicators comparing baseline vs. optimal trajectories

You have access to the ODE's underlying mathematical derivatives (Driving Forces). Positive values indicate upward momentum; negative values indicate downward drag.

OUTPUT YOUR REPORT IN THE FOLLOWING EXACT FORMAT USING MARKDOWN:

### 1. Overview
Provide a 2-3 sentence overview containing the final Target vs Baseline GDP figures and the total number of intervention points.

### 2. Quarterly Interventions & Reasoning
Group the provided monthly interventions into Quarters (e.g. "Q1 (Months 3-9)") in a MARKDOWN TABLE.
The table MUST have the following columns: [Quarter/Months], [Interventions], [Reasoning (Citing Driving Forces)].
Keep the reasoning concise but cite the positive/negative forces.

### 3. Economic Impact
YOU MUST USE A MARKDOWN TABLE. You MUST explicitly show both "Final Real GDP (Base 2011)" AND "Final Nominal GDP (Current)" for both Baseline and Optimal scenarios.

### 4. High-Level Strategy Summary
Provide a summary of the key interventions and their overall macroeconomic effects in a high-level way that anyone (a non-economist) can easily understand. Explain the "story" of how the levers guided the economy.

Do NOT output long textual bullet strings for every single month. Stick strictly to this structured format."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        self.mock_mode = True

        if api_key and OPENAI_AVAILABLE:
            try:
                # Rerouted to Hugging Face Free Serverless Inference
                self.client = OpenAI(
                    base_url="https://router.huggingface.co/v1/",
                    api_key=api_key 
                )
                self.mock_mode = False
                logger.info("AnalystAgent: LLM mode (Llama-3.3-70B-Instruct)")
            except Exception as e:
                logger.warning(f"AnalystAgent: OpenAI init failed ({e}), using mock mode")
        else:
            logger.info("AnalystAgent: Mock mode (template report)")

    def generate_report(
        self,
        objective: PolicyObjective,
        best_policy: Dict[int, Dict[str, float]],
        baseline_df: pd.DataFrame,
        optimal_df: pd.DataFrame,
        architect: ArchitectAgent,
        driving_forces: dict,
    ) -> str:
        """
        Generate an executive policy report.

        Args:
            objective:      The parsed PolicyObjective
            best_policy:    Dict of {month: {lever: value}} interventions
            baseline_df:    Baseline (no-intervention) trajectory DataFrame
            optimal_df:     Optimal trajectory DataFrame
            architect:      ArchitectAgent (for date/feature metadata)
            driving_forces: Extracted underlying mathematical velocities from ODE

        Returns:
            Markdown-formatted report string
        """
        logger.info("AnalystAgent: Generating executive report...")

        # ------ Build structured data summary ------
        summary_data = self._build_summary(
            objective, best_policy, baseline_df, optimal_df, architect, driving_forces
        )

        if self.mock_mode:
            return self._template_report(summary_data, objective, best_policy,
                                         baseline_df, optimal_df, architect)

        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(summary_data, indent=2)},
                ],
                temperature=0.3,
                max_tokens=3000,
            )

            report = response.choices[0].message.content.strip()
            logger.info("  Report generated via LLM")
            return report

        except Exception as e:
            logger.warning(f"  LLM call failed ({e}), using template report")
            return self._template_report(summary_data, objective, best_policy,
                                         baseline_df, optimal_df, architect)

    def _build_summary(
        self, objective, best_policy, baseline_df, optimal_df, architect, driving_forces
    ) -> dict:
        """Build a structured dictionary summarising the optimization results."""
        # Filter for the target year
        base_year_df = baseline_df[baseline_df.index.year == objective.target_year]
        opt_year_df = optimal_df[optimal_df.index.year == objective.target_year]

        # Fallbacks in case the year isn't fully in the dataframe
        if base_year_df.empty: base_year_df = baseline_df.iloc[-12:]
        if opt_year_df.empty: opt_year_df = optimal_df.iloc[-12:]

        summary = {
            "objective": {
                "target_gdp_trillions": objective.target_gdp_trillions,
                "target_year": objective.target_year,
                "max_inflation_pct": objective.max_inflation_pct,
                "max_unemployment_pct": objective.max_unemployment_pct,
                "allowed_levers": objective.allowed_levers,
            },
            "baseline": {
                "final_gdp_crore": float(base_year_df["Real_GDP_Crore"].sum()),
                "final_real_gdp_trillions_usd": float(
                    base_year_df["Real_GDP_Crore"].sum() / USD_TO_CRORE_FACTOR
                ),
                "final_nominal_gdp_trillions_usd": float(
                    (base_year_df["Real_GDP_Crore"].sum() / USD_TO_CRORE_FACTOR) * REAL_TO_NOMINAL_MULTIPLIER
                ),
                "avg_inflation": float(
                    base_year_df["Inflation_Rate_Monthly_RBI"].mean()
                ) if "Inflation_Rate_Monthly_RBI" in base_year_df.columns else None,
                "avg_unemployment": float(
                    base_year_df["Urban_Youth_Unemployment_Rate"].mean()
                ) if "Urban_Youth_Unemployment_Rate" in base_year_df.columns else None,
            },
            "optimal": {
                "final_gdp_crore": float(opt_year_df["Real_GDP_Crore"].sum()),
                "final_real_gdp_trillions_usd": float(
                    opt_year_df["Real_GDP_Crore"].sum() / USD_TO_CRORE_FACTOR
                ),
                "final_nominal_gdp_trillions_usd": float(
                    (opt_year_df["Real_GDP_Crore"].sum() / USD_TO_CRORE_FACTOR) * REAL_TO_NOMINAL_MULTIPLIER
                ),
                "avg_inflation": float(
                    opt_year_df["Inflation_Rate_Monthly_RBI"].mean()
                ) if "Inflation_Rate_Monthly_RBI" in opt_year_df.columns else None,
                "avg_unemployment": float(
                    opt_year_df["Urban_Youth_Unemployment_Rate"].mean()
                ) if "Urban_Youth_Unemployment_Rate" in opt_year_df.columns else None,
            },
            "policy_timeline_in_months": [
                f"Month {month} ({(architect.last_date + relativedelta(months=month)).strftime('%Y-%b')}): " + 
                ", ".join([f"{k} = {v:.4f}" for k, v in interventions.items()])
                for month, interventions in sorted(best_policy.items())
            ],
            "driving_forces": driving_forces,
        }
        return summary

    def _template_report(
        self, summary, objective, best_policy, baseline_df, optimal_df, architect
    ) -> str:
        """Generate a structured template report without an LLM."""

        baseline_real_gdp = summary["baseline"]["final_real_gdp_trillions_usd"]
        optimal_real_gdp = summary["optimal"]["final_real_gdp_trillions_usd"]
        real_gdp_delta = optimal_real_gdp - baseline_real_gdp

        baseline_nom_gdp = summary["baseline"]["final_nominal_gdp_trillions_usd"]
        optimal_nom_gdp = summary["optimal"]["final_nominal_gdp_trillions_usd"]
        nom_gdp_delta = optimal_nom_gdp - baseline_nom_gdp

        # Build policy roadmap table
        policy_rows = []
        for month, interventions in sorted(best_policy.items()):
            date = architect.last_date + relativedelta(months=month)
            for lever, value in interventions.items():
                lever_label = lever.replace("_", " ")
                policy_rows.append(
                    f"| {date.strftime('%B %Y')} | {lever_label} | {value:.2f} |"
                )

        policy_table = "\n".join(policy_rows) if policy_rows else "| — | — | — |"

        # Build comparison table
        avg_infl_base = summary["baseline"]["avg_inflation"]
        avg_infl_opt = summary["optimal"]["avg_inflation"]
        avg_unemp_base = summary["baseline"]["avg_unemployment"]
        avg_unemp_opt = summary["optimal"]["avg_unemployment"]

        report = f"""# VISIONARY — Optimal Economic Policy Report

## Summary

The Visionary optimization engine analyzed **{len(best_policy)} intervention \
points** across a **{len(optimal_df)-1}-month** forecast horizon to achieve a \
target GDP of **${objective.target_gdp_trillions:.1f} Trillion** by \
**{objective.target_year}**.

- **Baseline Nominal GDP** (no intervention): **${baseline_nom_gdp:.2f}T**
- **Optimal Nominal GDP** (with policy):      **${optimal_nom_gdp:.2f}T**
- **Nominal GDP uplift from policy**:         **${nom_gdp_delta:+.2f}T**

---

## Optimal Policy Roadmap

| Date | Lever | Value |
|------|-------|-------|
{policy_table}

---

## Projected Economic Impact

| Metric | Baseline | Optimal | Delta |
|--------|----------|---------|-------|
| Final Real GDP (Base 2011) | ${baseline_real_gdp:.2f}T | ${optimal_real_gdp:.2f}T | ${real_gdp_delta:+.2f}T |
| Final Nominal GDP (Current) | ${baseline_nom_gdp:.2f}T | ${optimal_nom_gdp:.2f}T | ${nom_gdp_delta:+.2f}T |
| Avg. Inflation Rate (%) | {avg_infl_base:.2f}% | {avg_infl_opt:.2f}% | {(avg_infl_opt - avg_infl_base):+.2f}% |
| Avg. Youth Unemployment (%) | {avg_unemp_base:.2f}% | {avg_unemp_opt:.2f}% | {(avg_unemp_opt - avg_unemp_base):+.2f}% |

---

## Secondary Effects & Spillovers

The policy lever adjustments propagate through the neural ODE's learned \
dynamics, affecting all 36 entangled macroeconomic variables simultaneously. \
Key secondary effects include changes in credit growth, foreign exchange \
reserves, and capital formation rates.

---

## Risks & Caveats

> **Model Limitations**: This forecast is based on a Neural ODE trained on \
170 months of historical data (June 2011 — July 2025). The model captures \
learned dynamics but cannot account for unprecedented structural shifts, \
global crises, or policy regime changes not present in the training data.

> **Policy Transmission**: Real-world policy transmission mechanisms involve \
lags, political constraints, and implementation challenges not modelled here. \
The recommended values represent mathematical optima, not politically \
calibrated targets.

> **Confidence Interval**: The deterministic nature of the Neural ODE means \
a single trajectory is produced per policy set. No uncertainty quantification \
is provided. Ensemble methods or Bayesian extensions are recommended for \
production policy-making.

---

*Report generated by VISIONARY Agentic Framework — {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
        logger.info("  Report generated via template")
        return report


# ==============================================================================
# ORCHESTRATOR — Pipeline
# ==============================================================================

class VisionaryOrchestrator:
    """
    Connects all 4 agents into a single .run(user_prompt) pipeline.

    Pipeline:
      1. StrategistAgent parses user goal → PolicyObjective
      2. OptimizerAgent searches for optimal policy using ArchitectAgent
      3. AnalystAgent generates executive report

    Usage:
        orchestrator = VisionaryOrchestrator(
            model_path="visionary_ode_model.pth",
            scaler_path="visionary_scaler.pkl",
            data_path="Final_Visionary_Economy_Dataset_Prepared.csv",
            openai_api_key=None,  # or "sk-..."
        )
        report = orchestrator.run("Achieve $10T GDP by 2035 keeping inflation under 6%")
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        data_path: str,
        openai_api_key: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            model_path:     Path to visionary_ode_model.pth
            scaler_path:    Path to visionary_scaler.pkl
            data_path:      Path to Final_Visionary_Economy_Dataset_Prepared.csv
            openai_api_key: OpenAI API key (None = mock mode for Agents 1 & 4)
            output_dir:     Directory to save output CSVs (None = same as model dir)
        """
        logger.info("=" * 72)
        logger.info("VISIONARY — Multi-Agent Orchestrator Initialising")
        logger.info("=" * 72)

        self.output_dir = output_dir or os.path.dirname(model_path) or "."

        # Initialise the 4 agents
        try:
            self.architect = ArchitectAgent(
                model_path=model_path,
                scaler_path=scaler_path,
                data_path=data_path,
            )
        except Exception as e:
            logger.critical(f"ArchitectAgent initialisation failed: {e}")
            raise RuntimeError(f"Cannot start orchestrator: ArchitectAgent failed — {e}")

        self.strategist = StrategistAgent(api_key=openai_api_key)
        self.analyst = AnalystAgent(api_key=openai_api_key)

        logger.info("Orchestrator: All agents initialised.")
        logger.info("=" * 72)

    def run(
        self,
        user_prompt: str,
        n_trials: int = 500,
        progress_callback=None,
        target_month: Optional[str] = None,
        user_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[str, optuna.Study]:
        """
        Execute the full Visionary pipeline.

        Args:
            user_prompt: Natural language policy goal
            n_trials:    Number of Optuna optimization trials
            target_month: Optional future date to start from
            user_overrides: Optional dictionary of { "Feature": {"t": val, "t_minus_1": val} } overrides

        Returns:
            Markdown-formatted executive policy report
        """
        logger.info("")
        logger.info("=" * 72)
        logger.info("VISIONARY — Pipeline Starting")
        logger.info(f"  Prompt: \"{user_prompt}\"")
        logger.info("=" * 72)

        pipeline_start = time.time()
        
        # Determine overridden state if partial tracking is requested
        initial_state_override = None
        if target_month and user_overrides:
            initial_state_override = self.architect.assimilate_partial_state(
                target_month=target_month,
                user_overrides=user_overrides
            )
            # Update last_date for charting and reporting purposes downstream
            self.architect.last_date = pd.to_datetime(target_month)

        # ============================================================
        # STEP 1: Strategist parses the goal
        # ============================================================
        try:
            objective = self.strategist.parse_goal(user_prompt)
        except Exception as e:
            logger.error(f"Step 1 (Strategist) failed: {e}")
            logger.info("  Using default PolicyObjective")
            objective = PolicyObjective()

        logger.info(f"  ✓ Step 1 complete: GDP=${objective.target_gdp_trillions}T by "
                    f"{objective.target_year}")

        # ============================================================
        # STEP 2: Optimizer searches for the best policy
        # ============================================================
        try:
            optimizer = OptimizerAgent(
                architect=self.architect,
                objective=objective,
                initial_state_override=initial_state_override,
            )
            best_policy, optimal_trajectory, baseline_trajectory, study = optimizer.optimize(
                n_trials=n_trials,
                progress_callback=progress_callback
            )
        except Exception as e:
            logger.error(f"Step 2 (Optimizer) failed: {e}")
            raise RuntimeError(f"Optimization failed: {e}")

        logger.info("  ✓ Step 2 complete: Optimal policy found")

        # ============================================================
        # STEP 3: Analyst generates the report
        # ============================================================
        try:
            state_to_use = initial_state_override if initial_state_override is not None else self.architect.get_initial_state()
            driving_forces = self.architect.get_driving_forces(state_to_use)
            report = self.analyst.generate_report(
                objective=objective,
                best_policy=best_policy,
                baseline_df=baseline_trajectory,
                optimal_df=optimal_trajectory,
                architect=self.architect,
                driving_forces=driving_forces,
            )
        except Exception as e:
            logger.error(f"Step 3 (Analyst) failed: {e}")
            report = f"# Report Generation Failed\n\nError: {e}"

        logger.info("  ✓ Step 3 complete: Report generated")

        # ============================================================
        # STEP 4: Save outputs
        # ============================================================
        try:
            baseline_path = os.path.join(self.output_dir, "baseline_trajectory.csv")
            optimal_path = os.path.join(self.output_dir, "optimal_trajectory.csv")
            report_path = os.path.join(self.output_dir, "visionary_report.md")

            baseline_trajectory.to_csv(baseline_path)
            optimal_trajectory.to_csv(optimal_path)
            with open(report_path, "w") as f:
                f.write(report)

            logger.info(f"  Saved: {baseline_path}")
            logger.info(f"  Saved: {optimal_path}")
            logger.info(f"  Saved: {report_path}")
        except Exception as e:
            logger.warning(f"  Failed to save outputs: {e}")

        # ============================================================
        # DONE
        # ============================================================
        elapsed = time.time() - pipeline_start
        logger.info("")
        logger.info("=" * 72)
        logger.info(f"VISIONARY — Pipeline Complete ({elapsed:.1f}s)")
        logger.info("=" * 72)

        return report, study


# ==============================================================================
# ENTRY POINT — Example usage
# ==============================================================================

if __name__ == "__main__":
    # ---- Configuration ----
    # Update these paths for your environment.
    # For Kaggle: the model and scaler are in /kaggle/working/ after training.
    MODEL_PATH = "visionary_ode_model.pth"
    SCALER_PATH = "visionary_scaler.pkl"
    DATA_PATH = "Visionary_Production_Dataset.csv"

    # Set your OpenAI API key here, or leave as None for mock mode.
    OPENAI_API_KEY = "hf_YOUR_NEW_SECURE_TOKEN_HERE"

    # ---- Run the pipeline ----
    orchestrator = VisionaryOrchestrator(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        data_path=DATA_PATH,
        openai_api_key=OPENAI_API_KEY,
    )

    report = orchestrator.run(
        user_prompt="Find a path to $3.5 Trillion Real GDP by 2028 keeping inflation under 6.0%.",
        n_trials=500,
    )

    print("\n" + report)
