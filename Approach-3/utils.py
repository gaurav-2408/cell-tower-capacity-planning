"""
utils.py
Data loading, cleaning, and derived feature engineering based on user's attribute list.
"""

import pandas as pd
import numpy as np

NUMERIC_DEFAULTS = {
    # defaults to avoid crashes; NaNs will be handled downstream
    "dl_bitrate": np.nan,
    "ul_bitrate": np.nan,
    "samples": np.nan,
    "rsrp_variance": np.nan,
    "throughput_degradation": np.nan,
    "range": np.nan,
    "current_usage": np.nan,
    "theoretical_capacity": np.nan,
    "actual_throughput": np.nan,
    "expected_throughput": np.nan,
    "active_connections": np.nan,
    "capacity": np.nan,
    "cell_density": np.nan,
    "distance": np.nan,
    "rsrp": np.nan,
    "rsrq": np.nan,
    "rssi": np.nan,
    "rscp": np.nan,
    "rxlev": np.nan,
    "rssnr": np.nan,
    "snr": np.nan,
    "cqi": np.nan,
    "throughput": np.nan,
    "velocity": np.nan,
    "peak_hour_congestion": np.nan,
    "weekend_vs_weekday_load": np.nan,
    "load_variability": np.nan,
    "operator_market_share": np.nan,
    "competitor_tower_density": np.nan,
    "congestion_frequency": np.nan,
    "peak_load_ratio": np.nan,
    "user_density": np.nan,
    "revenue_potential": np.nan,
    "overlap_coefficient": np.nan,
    "infrastructure_age": np.nan,
    "maintenance_cost": np.nan,
    "coordinate_accuracy": np.nan,
    "measurement_confidence": np.nan,
    "data_completeness": np.nan,
}

DERIVED_FEATURES = [
    "traffic_density",
    "load_factor",
    "performance_ratio",
    "congestion_score",
    "coverage_efficiency",
    "resource_utilization",
]

def load_data(path="data/towers.csv", dtypes=None):
    """
    Load CSV to DataFrame with safe defaults for missing numeric columns.
    """
    df = pd.read_csv(path)
    # Ensure numeric columns exist
    for col, default in NUMERIC_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df

def compute_derived_features(df, weights=None):
    """
    Compute derived features described by user.
    The function is robust to missing inputs: if inputs missing, derived set to NaN.
    Input:
        df: DataFrame with columns from user's list
        weights: dict for congestion_score weighting, default if None
    Returns DataFrame with new columns appended.
    """
    df = df.copy()
    # safe conversions
    for col in NUMERIC_DEFAULTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- traffic_density = samples / range^2 (avoid divide by zero) ---
    df["traffic_density"] = np.where(
        (df["range"].notnull()) & (df["range"] > 0),
        df["samples"] / (df["range"] ** 2),
        np.nan,
    )

    # --- load_factor = current_usage / theoretical_capacity ---
    df["load_factor"] = np.where(
        (df["theoretical_capacity"].notnull()) & (df["theoretical_capacity"] > 0),
        df["current_usage"] / df["theoretical_capacity"],
        np.nan,
    )

    # --- performance_ratio = actual_throughput / expected_throughput ---
    df["performance_ratio"] = np.where(
        (df["expected_throughput"].notnull()) & (df["expected_throughput"] > 0),
        df["actual_throughput"] / df["expected_throughput"],
        np.nan,
    )

    # --- coverage_efficiency = samples / range ---
    df["coverage_efficiency"] = np.where(
        (df["range"].notnull()) & (df["range"] > 0),
        df["samples"] / df["range"],
        np.nan,
    )

    # --- resource_utilization = active_connections / capacity ---
    df["resource_utilization"] = np.where(
        (df["capacity"].notnull()) & (df["capacity"] > 0),
        df["active_connections"] / df["capacity"],
        np.nan,
    )

    # --- congestion_score: weighted combination of load metrics ---
    # default weights if not provided
    if weights is None:
        weights = {
            "load_factor": 0.4,
            "traffic_density": 0.25,
            "throughput_degradation": 0.15,
            "resource_utilization": 0.2,
        }
    # normalize inputs to 0..1 then weighted sum
    def minmax_series(s):
        if s.dropna().empty:
            return s * 0.0 + np.nan
        mn, mx = s.min(), s.max()
        if mn == mx:
            return (s - mn) * 0.0
        return (s - mn) / (mx - mn)

    comps = {}
    for k in ["load_factor", "traffic_density", "throughput_degradation", "resource_utilization"]:
        if k in df.columns:
            comps[k] = minmax_series(df[k].fillna(np.nan))
        else:
            comps[k] = pd.Series(np.nan, index=df.index)

    df["congestion_score"] = (
        weights.get("load_factor", 0) * comps["load_factor"]
        + weights.get("traffic_density", 0) * comps["traffic_density"]
        + weights.get("throughput_degradation", 0) * comps["throughput_degradation"]
        + weights.get("resource_utilization", 0) * comps["resource_utilization"]
    )

    # If all NaN, keep as NaN
    df.loc[df[["load_factor","traffic_density","throughput_degradation","resource_utilization"]].isna().all(axis=1), "congestion_score"] = np.nan

    # --- other light derived: signal_quality_ranking (example composite) ---
    # combine rsrp, rsrq, rssnr into a ranking (higher better)
    components = []
    for s in ["rsrp", "rsrq", "rssnr"]:
        if s in df.columns:
            components.append(minmax_series(df[s].fillna(np.nan)))
    if components:
        df["signal_quality_ranking"] = pd.concat(components, axis=1).mean(axis=1)
    else:
        df["signal_quality_ranking"] = np.nan

    # TTL: Keep list of derived features
    df["derived_features_list"] = ", ".join(DERIVED_FEATURES)

    return df

def prepare_features_for_clustering(df, features=None, fill_strategy="median"):
    """
    Selects features, handles missing values and returns X, scaler-ready df.
    features: list of feature names to use for clustering. If None, pick sensible defaults.
    fill_strategy: median/mean/zero
    """
    default_features = [
        "traffic_density", "load_factor", "performance_ratio", "congestion_score",
        "coverage_efficiency", "resource_utilization",
        "rsrp", "rsrq", "rssnr", "cqi", "throughput", "load_variability",
        "operator_market_share", "competitor_tower_density", "cell_density",
        "user_density", "peak_load_ratio", "distance"
    ]
    if features is None:
        features = [f for f in default_features if f in df.columns]

    X = df[features].copy()
    if fill_strategy == "median":
        X = X.fillna(X.median())
    elif fill_strategy == "mean":
        X = X.fillna(X.mean())
    else:
        X = X.fillna(0)

    # If any column is constant (zero variance), leave it -- scaler will handle
    return X, features
