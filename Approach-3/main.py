import pandas as pd
from data_utils import load_and_clean, compute_cell_density, compute_overlap_coefficient, compute_peak_hour_and_trends
from clustering_models import cluster_geo_kmeans, cluster_geo_dbscan, cluster_features_kmeans
from evaluation import cluster_summary
from visualization import plot_hexbin, plot_folium_clusters

def main():
    print("Loading and cleaning data...")
    df = load_and_clean("dataset.csv")

    print("Computing cell density (500m)...")
    df = compute_cell_density(df, radius_meters=500)

    print("Computing overlap coefficient...")
    df = compute_overlap_coefficient(df)

    print("Computing peak hour & congestion trends...")
    df = compute_peak_hour_and_trends(df, tower_id_col="tower_cell_id")

    # --- Geo KMeans ---
    print("Running geographic KMeans clustering...")
    df_geo_k, km_geo = cluster_geo_kmeans(df, n_clusters=5)
    df_geo_k.to_csv("clustered_geo_kmeans.csv", index=False)

    # --- Geo DBSCAN (on sample) ---
    print("Running geographic DBSCAN clustering (sampled)...")
    df_geo_d, db_geo = cluster_geo_dbscan(df_geo_k, eps=0.06, min_samples=10, sample_size=10000)
    df_geo_d.to_csv("clustered_geo_dbscan_sample.csv", index=False)

    # --- Throughput KMeans ---
    print("Running throughput KMeans clustering...")
    feat_throughput = ["dl_bitrate", "ul_bitrate", "throughput", "traffic_density"]
    feat_throughput = [f for f in feat_throughput if f in df_geo_d.columns]

    df_th, km_thr = cluster_features_kmeans(df_geo_d, feature_list=feat_throughput, n_clusters=5)
    df_th.to_csv("clustered_throughput_kmeans.csv", index=False)

    # --- Summaries ---
    print("Saving cluster summaries...")
    summary_geo = cluster_summary(df_geo_k, "geo_kmeans_label", ["tower_range", "traffic_density", "throughput"])
    summary_geo.to_csv("geo_kmeans_summary.csv")

    summary_thr = cluster_summary(df_th, "feature_kmeans_label", feat_throughput)
    summary_thr.to_csv("throughput_kmeans_summary.csv")

    # --- Visualizations ---
    print("Creating visualizations...")
    plot_hexbin(df, feature="throughput", save_path="hexbin_throughput.png")
    plot_folium_clusters(df_geo_k, label_col="geo_kmeans_label",
                         lat_col="servingcell_lat" if "servingcell_lat" in df.columns else "latitude",
                         lon_col="servingcell_lon" if "servingcell_lon" in df.columns else "longitude",
                         out_html="map_geo_kmeans.html", feature="throughput")

    print("Pipeline complete. Outputs saved.")

if __name__ == "__main__":
    main()
