"""Shared utilities for the manuscript revision experiments.

Provides a single, consistent data loader, feature engineering, evaluation harness
(K-fold cross-validation x random seeds), aspect-ratio stratification,
inference-latency benchmark, and noise-robustness study.

Used by:
  - notebooks/00_unified_forward_evaluation.ipynb
  - notebooks/01_physics_informed_ann.ipynb
  - notebooks/02_probabilistic_inverse.ipynb
"""
from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "Experimental_profiles_fitted.csv"
OUTPUTS = ROOT / "outputs"
FIGS = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
for d in (OUTPUTS, FIGS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Data loader (matches notebooks' format: every two columns is a sample)
# ----------------------------------------------------------------------------
@dataclass
class Dataset:
    X_raw: np.ndarray            # (N, 4)  columns: [P, SOD, V, N]
    Y: np.ndarray                # (N, L)  cross-sectional depth profile
    x_positions: np.ndarray      # (L,)    common x grid
    feature_names_raw: list[str] = field(default_factory=lambda: ["P", "SOD", "V", "N"])

    @property
    def n_samples(self) -> int:
        return self.X_raw.shape[0]

    @property
    def n_features_raw(self) -> int:
        return self.X_raw.shape[1]

    @property
    def n_profile_points(self) -> int:
        return self.Y.shape[1]


def load_dataset(csv_path: Path | str = DATA_CSV) -> Dataset:
    raw = pd.read_csv(csv_path, header=None)
    num_cols = raw.shape[1]
    X_list, Y_list = [], []
    x_values = None
    for i in range(0, num_cols, 2):
        try:
            sod = float(raw.iloc[0, i])
            p   = float(raw.iloc[1, i])
            v   = float(raw.iloc[2, i])
            n   = float(raw.iloc[3, i])
            y_vals = raw.iloc[4:, i + 1].values.astype(float)
            if np.isnan(y_vals).any() or np.all(y_vals == 0):
                continue
            X_list.append([p, sod, v, n])
            Y_list.append(y_vals)
            if x_values is None:
                x_values = raw.iloc[4:, i].values.astype(float)
        except Exception:
            continue
    X = np.asarray(X_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)
    x_positions = np.asarray(x_values, dtype=float)
    return Dataset(X_raw=X, Y=Y, x_positions=x_positions)


# ----------------------------------------------------------------------------
# Physics-informed feature engineering (applied uniformly to all models)
# ----------------------------------------------------------------------------
PHYSICS_FEATURE_NAMES = [
    "P", "SOD", "V", "N",                # raw
    "P*N",         # cumulative erosive potential
    "SOD/V",       # exposure / dwell time per unit length
    "P/V",         # erosive intensity per unit traverse
    "log(N)",      # diminishing-returns of additional passes
    "P*N/V",       # specific cumulative dose
]


def add_physics_features(X_raw: np.ndarray) -> np.ndarray:
    P, SOD, V, N = X_raw[:, 0], X_raw[:, 1], X_raw[:, 2], X_raw[:, 3]
    eps = 1e-6
    X_eng = np.column_stack([
        P, SOD, V, N,
        P * N,
        SOD / (V + eps),
        P / (V + eps),
        np.log(np.clip(N, 1e-3, None)),
        P * N / (V + eps),
    ])
    return X_eng


# ----------------------------------------------------------------------------
# Aspect-ratio stratification helpers
# ----------------------------------------------------------------------------
def channel_depth_width(profile: np.ndarray, x_positions: np.ndarray) -> tuple[float, float]:
    """Depth = |min(profile)|; width at half-max-depth."""
    depth = float(np.min(profile))
    if depth >= 0:
        return 0.0, 0.0
    thr = 0.5 * depth
    idx = np.where(profile <= thr)[0]
    if len(idx) > 0:
        width = float(x_positions[idx[-1]] - x_positions[idx[0]])
    else:
        width = float(x_positions[-1] - x_positions[0])
    return abs(depth), max(width, 1e-9)


def aspect_ratios(Y: np.ndarray, x_positions: np.ndarray) -> np.ndarray:
    out = np.empty(Y.shape[0])
    for i, prof in enumerate(Y):
        d, w = channel_depth_width(prof, x_positions)
        out[i] = d / w if w > 0 else 0.0
    return out


def ar_bins(ar: np.ndarray) -> np.ndarray:
    """3 bins: low (<0.5), medium (0.5-1.5), high (>=1.5). Returns string labels."""
    labels = np.empty(len(ar), dtype=object)
    labels[ar < 0.5] = "low (AR<0.5)"
    labels[(ar >= 0.5) & (ar < 1.5)] = "medium (0.5\u2264AR<1.5)"
    labels[ar >= 1.5] = "high (AR\u22651.5)"
    return labels


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2":  float(r2_score(y_true, y_pred)),
    }


def metrics_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Per-sample metrics for stratified analysis."""
    mae = np.mean(np.abs(y_true - y_pred), axis=1)
    mse = np.mean((y_true - y_pred) ** 2, axis=1)
    return {"MAE": mae, "MSE": mse}


# ----------------------------------------------------------------------------
# Cross-validation harness
# ----------------------------------------------------------------------------
ModelFactory = Callable[[int, int], object]   # (n_in, n_out) -> fresh estimator


def cross_validate_model(
    model_factory: ModelFactory,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_splits: int = 5,
    seeds: Iterable[int] = (0, 1, 2),
    scale_X: bool = True,
    scale_Y: bool = False,
    fit_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Run k-fold cross validation across multiple seeds.

    Returns a long-form DataFrame with one row per (seed, fold).
    `model_factory(n_in, n_out)` must return an unfit estimator with sklearn-style
    `.fit(X, y)` and `.predict(X)`.
    """
    fit_kwargs = fit_kwargs or {}
    rows: list[dict] = []
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(kf.split(X)):
            X_tr, X_te = X[tr], X[te]
            Y_tr, Y_te = Y[tr], Y[te]
            sx = StandardScaler().fit(X_tr) if scale_X else None
            sy = StandardScaler().fit(Y_tr) if scale_Y else None
            X_tr_s = sx.transform(X_tr) if sx else X_tr
            X_te_s = sx.transform(X_te) if sx else X_te
            Y_tr_s = sy.transform(Y_tr) if sy else Y_tr
            est = model_factory(X_tr_s.shape[1], Y_tr_s.shape[1])
            est.fit(X_tr_s, Y_tr_s, **fit_kwargs)
            Y_pred_s = np.asarray(est.predict(X_te_s))
            Y_pred = sy.inverse_transform(Y_pred_s) if sy else Y_pred_s
            m = regression_metrics(Y_te, Y_pred)
            rows.append({"seed": seed, "fold": fold, **m})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict[str, float]:
    """Mean and std of MSE/MAE/R2 across rows."""
    out = {}
    for k in ("MSE", "MAE", "R2"):
        out[f"{k}_mean"] = float(df[k].mean())
        out[f"{k}_std"]  = float(df[k].std(ddof=1))
    return out


# ----------------------------------------------------------------------------
# AR-stratified evaluation (single train/test split, per-bin metrics)
# ----------------------------------------------------------------------------
def ar_stratified_metrics(
    Y_true: np.ndarray, Y_pred: np.ndarray, x_positions: np.ndarray
) -> pd.DataFrame:
    ar = aspect_ratios(Y_true, x_positions)
    bins = ar_bins(ar)
    per = metrics_per_sample(Y_true, Y_pred)
    df = pd.DataFrame({"AR": ar, "bin": bins, **per})
    rows = []
    for b in ("low (AR<0.5)", "medium (0.5\u2264AR<1.5)", "high (AR\u22651.5)"):
        sub = df[df["bin"] == b]
        if len(sub) == 0:
            rows.append({"bin": b, "n": 0, "MAE_mean": np.nan,
                         "MAE_std": np.nan, "MSE_mean": np.nan})
            continue
        rows.append({
            "bin": b,
            "n": int(len(sub)),
            "MAE_mean": float(sub["MAE"].mean()),
            "MAE_std":  float(sub["MAE"].std(ddof=1)) if len(sub) > 1 else 0.0,
            "MSE_mean": float(sub["MSE"].mean()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Inference latency benchmark
# ----------------------------------------------------------------------------
def measure_inference_latency(predict_fn: Callable[[np.ndarray], np.ndarray],
                              X_sample: np.ndarray,
                              *, n_warmup: int = 3, n_repeat: int = 30) -> dict[str, float]:
    """Per-sample inference latency in milliseconds (single-sample batches)."""
    x1 = X_sample[:1]
    for _ in range(n_warmup):
        predict_fn(x1)
    times_ms = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        predict_fn(x1)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times_ms)
    return {"latency_ms_median": float(np.median(arr)),
            "latency_ms_mean":   float(arr.mean()),
            "latency_ms_std":    float(arr.std(ddof=1))}


# ----------------------------------------------------------------------------
# Noise-robustness study
# ----------------------------------------------------------------------------
def noise_robustness_curve(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    *,
    sigmas: Iterable[float] = (0.0, 0.01, 0.02, 0.05, 0.10),
    n_repeats: int = 20,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Inject zero-mean Gaussian noise of relative std `sigma` into each input
    feature (sigma is relative to that feature's std across the test set).
    Returns mean +/- std of R2/MAE/MSE over `n_repeats` realizations per sigma.
    """
    rng = np.random.default_rng(rng_seed)
    feat_std = np.std(X_test, axis=0)
    rows = []
    for sigma in sigmas:
        m_list = []
        for _ in range(n_repeats):
            noise = rng.normal(0.0, sigma * feat_std, size=X_test.shape) if sigma > 0 else 0.0
            X_noisy = X_test + noise
            Y_pred = predict_fn(X_noisy)
            m_list.append(regression_metrics(Y_test, Y_pred))
        agg = {k: np.array([m[k] for m in m_list]) for k in m_list[0]}
        rows.append({
            "sigma": sigma,
            "MSE_mean": float(agg["MSE"].mean()), "MSE_std": float(agg["MSE"].std(ddof=1)),
            "MAE_mean": float(agg["MAE"].mean()), "MAE_std": float(agg["MAE"].std(ddof=1)),
            "R2_mean":  float(agg["R2"].mean()),  "R2_std":  float(agg["R2"].std(ddof=1)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Result persistence helpers
# ----------------------------------------------------------------------------
def save_results(name: str, summary: dict, cv_df: pd.DataFrame | None = None,
                 ar_df: pd.DataFrame | None = None,
                 noise_df: pd.DataFrame | None = None,
                 latency: dict | None = None) -> None:
    payload = {"name": name, "summary": summary}
    if latency is not None:
        payload["latency"] = latency
    (TABLES / f"{name}_summary.json").write_text(json.dumps(payload, indent=2))
    if cv_df is not None:
        cv_df.to_csv(TABLES / f"{name}_cv.csv", index=False)
    if ar_df is not None:
        ar_df.to_csv(TABLES / f"{name}_ar_strat.csv", index=False)
    if noise_df is not None:
        noise_df.to_csv(TABLES / f"{name}_noise.csv", index=False)
    print(f"  saved -> {TABLES / (name + '_*.{json,csv}')}")


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_global_seed(seed: int = 42) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        warnings.warn("TensorFlow not available; only numpy/random/PYTHONHASHSEED seeded")
