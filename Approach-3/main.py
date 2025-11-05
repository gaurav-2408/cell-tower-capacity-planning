import os
from scripts.data_utils import load_and_clean, compute_cell_density, compute_overlap_coefficient, compute_peak_hour_and_trends
from scripts.clustering_models import cluster_geo_kmeans, cluster_features_kmeans
from scripts.visualization import plot_hexbin, plot_folium_clusters
from scripts.report_generator import generate_html_report

INPUT_DIR = "input"
OUTPUT_DIR = "output"
DATA_FILE = os.path.join(INPUT_DIR, "dataset.csv")

def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading and cleaning data...")
    df = load_and_clean(DATA_FILE)

    print("Computing cell density (500m)...")
    df = compute_cell_density(df, radius_meters=500)

    print("Computing overlap coefficient...")
    df = compute_overlap_coefficient(df)

    print("Computing peak hour & congestion trends...")
    df = compute_peak_hour_and_trends(df)

    print("Running geographic KMeans clustering...")
    df_geo_k, km_geo = cluster_geo_kmeans(df, n_clusters=5)

    print("Running throughput KMeans clustering...")
    feat_throughput = ["dl_bitrate", "ul_bitrate", "throughput", "traffic_density"]
    df_th, km_thr = cluster_features_kmeans(df_geo_k, feature_list=feat_throughput, n_clusters=5)

    print("Generating hexbin plot...")
    plot_hexbin(
    df_th,
    lat_col="latitude",
    lon_col="longitude",
    feature="throughput",
    output_file=os.path.join(OUTPUT_DIR, "hexbin_throughput.png")
    )
    print("Generating geographic clusters map...")
    plot_folium_clusters(
        df_geo_k,
        label_col="geo_kmeans_label",
        output_file=os.path.join(OUTPUT_DIR, "map_geo_kmeans.html"),
    )

    print("Generating HTML report...")
    generate_html_report(df_geo_k, os.path.join(OUTPUT_DIR, "final_report.html"))

    print("Done. Outputs saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
