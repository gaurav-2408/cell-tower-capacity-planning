import folium
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import os

def plot_hexbin(df, lat_col="latitude", lon_col="longitude", feature="throughput", gridsize=60, cmap="YlOrRd", output_file="output/hexbin_geo_throughput.png"):
    if lat_col not in df.columns or lon_col not in df.columns:
        print("Latitude/Longitude columns missing.")
        return
    if feature not in df.columns:
        print("Feature column missing.")
        return

    df = df.dropna(subset=[lat_col, lon_col, feature])
    if df.empty:
        print("No valid data points for plotting.")
        return

    plt.figure(figsize=(8, 6))
    plt.hexbin(df[lon_col], df[lat_col], C=df[feature], gridsize=50, cmap=cmap, reduce_C_function=np.mean, mincnt=1)
    plt.colorbar(label=f"Avg {feature}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Hexbin of {feature}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Hexbin geo plot saved: {output_file}")


def plot_folium_clusters(df, lat_col="servingcell_lat", lon_col="servingcell_lon", label_col="geo_kmeans_label", output_file="output/map_geo_kmeans.html"):
    if df.empty or lat_col not in df.columns or lon_col not in df.columns:
        print("No coordinates available for plotting.")
        return
    m = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()], zoom_start=7)
    boundary_group = folium.FeatureGroup(name="Cluster Boundaries")
    for cluster_id in df[label_col].dropna().unique():
        cluster_points = df[df[label_col] == cluster_id][[lat_col, lon_col]].dropna().values
        for lat, lon in cluster_points:
            folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True, fill_opacity=0.6, popup=f"Cluster {cluster_id}").add_to(m)
        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points)
            boundary = cluster_points[hull.vertices]
            folium.Polygon(locations=[(p[0], p[1]) for p in boundary], color="red", weight=2, fill=False, tooltip=f"Cluster {cluster_id}").add_to(boundary_group)
    boundary_group.add_to(m)
    folium.LayerControl().add_to(m)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    m.save(output_file)
    print(f"Folium map saved: {output_file}")
