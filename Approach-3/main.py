import os
import pandas as pd

from scripts.data_utils import (
    load_and_clean,
    compute_cell_density,
    compute_overlap_coefficient,
    compute_peak_hour_and_trends,   # ✅ fixed import
)
from scripts.clustering_models import (
    cluster_geo_kmeans,
    cluster_features_kmeans,
    cluster_geo_dbscan,
)
from scripts.visualization import plot_hexbin, plot_folium_clusters
from scripts.evaluate_metrics import evaluate_metrics


# Folders
INPUT_DIR = "input"
OUTPUT_DIR = "output"
DATA_FILE = os.path.join(INPUT_DIR, "dataset.csv")


def evaluate_all_metrics(df, output_path):
    """Run clustering evaluation metrics and save results."""
    with open(output_path, "w", encoding="utf-8") as f:
        # --- Geographic KMeans ---
        if "geo_kmeans_label" in df.columns:
            X_geo = df[["servingcell_lat", "servingcell_lon"]].dropna().values
            labels_geo = df.loc[~df["geo_kmeans_label"].isna(), "geo_kmeans_label"].astype(int).values

            geo_results = evaluate_metrics(X_geo, labels_geo, sample_size=5000)
            f.write("\nGeographic KMeans Clusters\n")
            for k, v in geo_results.items():
                f.write(f"{k}: {v}\n")

        # --- Throughput KMeans ---
        if "feature_kmeans_label" in df.columns:
            features = ["dl_bitrate", "ul_bitrate", "throughput", "traffic_density"]
            X_thr = df[features].dropna().values
            labels_thr = df.loc[~df["feature_kmeans_label"].isna(), "feature_kmeans_label"].astype(int).values

            thr_results = evaluate_metrics(X_thr, labels_thr, sample_size=5000)
            f.write("\nThroughput KMeans Clusters\n")
            for k, v in thr_results.items():
                f.write(f"{k}: {v}\n")


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading and cleaning data...")
    df = load_and_clean(DATA_FILE)

    # Derived features
    print("Computing cell density (500m)...")
    df = compute_cell_density(df, radius_meters=500)

    print("Computing overlap coefficient...")
    df = compute_overlap_coefficient(df)

    print("Computing peak hour & congestion trends...")
    df = compute_peak_hour_and_trends(df)   # ✅ fixed function call

    # --- Clustering ---
    print("Running geographic KMeans clustering...")
    df_geo_k, km_geo = cluster_geo_kmeans(df, n_clusters=5)

    print("Running throughput KMeans clustering...")
    feat_throughput = ["dl_bitrate", "ul_bitrate", "throughput", "traffic_density"]
    df_th, km_thr = cluster_features_kmeans(df_geo_k, feature_list=feat_throughput, n_clusters=5)

    print("Running geographic DBSCAN clustering...")
    df_geo_d, db_geo = cluster_geo_dbscan(df_th, eps=0.06, min_samples=10)

    # --- Save outputs ---
    df_geo_k.to_csv(os.path.join(OUTPUT_DIR, "clustered_geo_kmeans.csv"), index=False)
    df_th.to_csv(os.path.join(OUTPUT_DIR, "clustered_throughput_kmeans.csv"), index=False)
    df_geo_d.sample(n=500, random_state=42).to_csv(
        os.path.join(OUTPUT_DIR, "clustered_geo_dbscan_sample.csv"), index=False
    )

    # --- Visualizations ---
    print("Generating hexbin plot (throughput)...")
    plot_hexbin(df_th, feature="throughput", output_file=os.path.join(OUTPUT_DIR, "hexbin_throughput.png"))

    print("Generating geographic clusters map...")
    plot_folium_clusters(
        df_geo_k,
        label_col="geo_kmeans_label",
        output_file=os.path.join(OUTPUT_DIR, "map_geo_kmeans.html"),
    )

    # --- Metrics ---
    print("Evaluating clustering performance...")
    metrics_output = os.path.join(OUTPUT_DIR, "metrics_results.txt")
    evaluate_all_metrics(df_th, metrics_output)

    print("Pipeline completed. Outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
