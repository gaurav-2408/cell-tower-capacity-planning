"""
main.py
Orchestration script to run Phase C pipeline:
1) Load data
2) Compute derived features
3) Prepare features for clustering
4) Run KMeans and DBSCAN
5) Evaluate and save outputs
6) Produce hexbin and folium maps
"""

import os
import pandas as pd
from scripts.utils import load_data, compute_derived_features, prepare_features_for_clustering
from scripts.clustering import run_kmeans, run_dbscan
from scripts.visualize import plot_hexbin, folium_cluster_map
from scripts.evaluate import silhouette_for_labels, cluster_summary
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/towers.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Computing derived features...")
    df = compute_derived_features(df)

    print("Selecting features for clustering and preparing X...")
    X, feature_list = prepare_features_for_clustering(df, features=None, fill_strategy="median")
    print("Using features:", feature_list)

    # Standard scaler (useful for silhouette)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- KMeans ---
    n_clusters = 5
    print(f"Running KMeans with k={n_clusters} ...")
    df_k = run_kmeans(df, X, n_clusters=n_clusters)
    df_k.to_csv(f"{OUT_DIR}/clusters_kmeans.csv", index=False)
    print("Saved clusters_kmeans.csv")

    # silhouette (note: X_scaled used)
    try:
        k_sil = silhouette_for_labels(X_scaled, df_k["kmeans_label"].values)
    except Exception:
        k_sil = None
    print("KMeans silhouette score:", k_sil)

    # --- DBSCAN ---
    print("Running DBSCAN (eps=0.5,min_samples=5) ...")
    df_db = run_dbscan(df, X, eps=0.5, min_samples=5)
    df_db.to_csv(f"{OUT_DIR}/clusters_dbscan.csv", index=False)
    print("Saved clusters_dbscan.csv")
    try:
        db_sil = silhouette_for_labels(X_scaled, df_db["dbscan_label"].values)
    except Exception:
        db_sil = None
    print("DBSCAN silhouette score:", db_sil)

    # --- Summaries ---
    print("Generating cluster summaries (KMeans)...")
    ks = cluster_summary(df_k, "kmeans_label", feature_list)
    ks.to_csv(f"{OUT_DIR}/kmeans_summary.csv")
    print("Saved kmeans_summary.csv")

    print("Generating cluster summaries (DBSCAN)...")
    ds = cluster_summary(df_db, "dbscan_label", feature_list)
    ds.to_csv(f"{OUT_DIR}/dbscan_summary.csv")
    print("Saved dbscan_summary.csv")

    # --- Visualizations ---
    print("Creating hexbin of congestion_score ...")
    plot_hexbin(df, c_col="congestion_score", gridsize=80, save_path=f"{OUT_DIR}/hexbin_congestion.png")

    print("Creating folium map for KMeans ...")
    folium_cluster_map(df_k, label_col="kmeans_label", congestion_col="congestion_score", out_html=f"{OUT_DIR}/map_kmeans.html", popup_cols=["cell_id","mcc","mnc"])

    print("Creating folium map for DBSCAN ...")
    folium_cluster_map(df_db, label_col="dbscan_label", congestion_col="congestion_score", out_html=f"{OUT_DIR}/map_dbscan.html", popup_cols=["cell_id","mcc","mnc"])

    print("All outputs saved in", OUT_DIR)

if __name__ == "__main__":
    main()
