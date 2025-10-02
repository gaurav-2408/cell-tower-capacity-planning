import folium
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import os


def plot_hexbin(
    df,
    x_col=None,
    y_col=None,
    feature=None,
    gridsize=50,
    cmap="Blues",
    output_file="output/hexbin_plot.png",
):
    """
    Create a hexbin plot.
    - If x_col and y_col are given: 2D hexbin (x vs y).
    - If feature is given: 1D hexbin-like plot (feature vs sample index).
    """
    plt.figure(figsize=(10, 6))

    if x_col and y_col:
        plt.hexbin(df[x_col], df[y_col], gridsize=gridsize, cmap=cmap, mincnt=1)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Hexbin plot of {x_col} vs {y_col}")
    elif feature:
        plt.hexbin(range(len(df)), df[feature], gridsize=gridsize, cmap=cmap, mincnt=1)
        plt.xlabel("Samples")
        plt.ylabel(feature)
        plt.title(f"Hexbin plot of {feature}")
    else:
        print("❌ Please provide either (x_col & y_col) or feature.")
        return

    plt.colorbar(label="Counts")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Hexbin plot saved: {output_file}")


def plot_folium_clusters(
    df,
    lat_col="servingcell_lat",
    lon_col="servingcell_lon",
    label_col="geo_kmeans_label",
    output_file="output/map_geo_kmeans.html",
):
    """
    Create a Folium map with all cluster boundaries drawn at once.
    Each cluster gets a convex hull polygon.
    Boundaries can be toggled via LayerControl.
    """
    if df.empty or lat_col not in df.columns or lon_col not in df.columns:
        print("No coordinates available for plotting.")
        return

    # Center map on dataset mean
    m = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()], zoom_start=7)

    # Add cluster boundaries as a separate FeatureGroup
    boundary_group = folium.FeatureGroup(name="Cluster Boundaries")

    # Loop over clusters
    for cluster_id in df[label_col].dropna().unique():
        cluster_points = df[df[label_col] == cluster_id][[lat_col, lon_col]].dropna().values

        # Add tower markers
        for lat, lon in cluster_points:
            folium.CircleMarker(
                [lat, lon],
                radius=2,
                color="blue",
                fill=True,
                fill_opacity=0.6,
                popup=f"Cluster {cluster_id}",
            ).add_to(m)

        # Draw convex hull for cluster boundary
        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points)
            boundary = cluster_points[hull.vertices]
            folium.Polygon(
                locations=[(p[0], p[1]) for p in boundary],
                color="red",
                weight=2,
                fill=False,
                tooltip=f"Cluster {cluster_id}",
            ).add_to(boundary_group)

    # Add boundaries group + control to map
    boundary_group.add_to(m)
    folium.LayerControl().add_to(m)

    # Save map
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    m.save(output_file)
    print(f"Folium map saved: {output_file}")
