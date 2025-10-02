import pandas as pd
import numpy as np

def load_and_clean(path="dataset.csv"):
    df = pd.read_csv(path, low_memory=False)

    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        except Exception:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    num_cols = [
        "rsrp", "rsrq", "snr", "cqi", "rssi",
        "dl_bitrate", "ul_bitrate",
        "tower_samples", "tower_range", "tower_average_signal",
        "servingcell_distance", "distance_to_tower", "speed",
        "latitude", "longitude", "tower_lat", "tower_lon",
        "servingcell_lat", "servingcell_lon"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mobility_pattern" in df.columns:
        df["mobility_category"] = pd.Series(dtype="object")
        df.loc[df["mobility_pattern"].str.contains("static|still|home|office", na=False), "mobility_category"] = "static"
        df.loc[df["mobility_pattern"].str.contains("move|walk|veh|transit", na=False), "mobility_category"] = "mobile"

    if "tower_samples" in df.columns and "tower_range" in df.columns:
        df["traffic_density"] = np.where(df["tower_range"] > 0,
                                         df["tower_samples"] / (df["tower_range"] ** 2), np.nan)
        df["coverage_efficiency"] = np.where(df["tower_range"] > 0,
                                             df["tower_samples"] / df["tower_range"], np.nan)

    if "dl_bitrate" in df.columns and "ul_bitrate" in df.columns:
        df["throughput"] = df["dl_bitrate"] + df["ul_bitrate"]

    return df


def compute_cell_density(df, radius_meters=500):
    if "tower_lat" not in df.columns or "tower_lon" not in df.columns:
        print("Tower coordinates missing. Skipping cell density.")
        return df

    from sklearn.neighbors import BallTree

    towers = df[["tower_lat", "tower_lon"]].dropna().drop_duplicates()
    pts = np.radians(towers[["tower_lat", "tower_lon"]].values)

    tree = BallTree(pts, metric="haversine")
    r = radius_meters / 6371000.0  # radius in radians
    counts = tree.query_radius(pts, r=r, count_only=True)

    towers["cell_density_500m"] = counts
    df = df.merge(towers, on=["tower_lat", "tower_lon"], how="left")

    return df


def compute_overlap_coefficient(df):
    if "cell_density_500m" not in df.columns:
        print("Cell density not computed. Skipping overlap coefficient.")
        return df

    df["overlap_coefficient"] = df["cell_density_500m"] / df["cell_density_500m"].max()
    return df


def compute_peak_hour_and_trends(df, tower_id_col="tower_cell_id"):
    if "timestamp" not in df.columns or tower_id_col not in df.columns:
        print("Timestamps or tower IDs missing. Skipping congestion trends.")
        return df

    df["hour"] = df["timestamp"].dt.hour
    peak_loads = df.groupby([tower_id_col, "hour"])["throughput"].mean().reset_index()
    peak_hours = peak_loads.loc[peak_loads.groupby(tower_id_col)["throughput"].idxmax()]
    peak_hours.rename(columns={"hour": "peak_hour_congestion"}, inplace=True)

    df = df.merge(peak_hours[[tower_id_col, "peak_hour_congestion"]], on=tower_id_col, how="left")

    return df
