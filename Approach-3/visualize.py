"""
visualize.py
Hexbin maps (matplotlib) and interactive Folium maps to visualize clusters and congestion risk.
"""

import matplotlib.pyplot as plt
import numpy as np
import folium
from folium.plugins import MarkerCluster

def plot_hexbin(df, lon_col="longitude", lat_col="latitude", c_col="congestion_score", gridsize=60, reduce_C_function=np.mean, save_path=None):
    """
    Simple hexbin: averages c_col in each hex.
    """
    plt.figure(figsize=(10, 6))
    hb = plt.hexbin(df[lon_col], df[lat_col], C=df[c_col], gridsize=gridsize, reduce_C_function=reduce_C_function, cmap="YlOrRd")
    plt.colorbar(hb, label=f"Avg {c_col}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Hexbin of {c_col}")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved hexbin to {save_path}")
    plt.show()

def folium_cluster_map(df, lat_col="latitude", lon_col="longitude", label_col="kmeans_label", congestion_col="congestion_score", out_html="map_clusters.html", popup_cols=None):
    """
    Create a folium map with MarkerCluster. Marker color derived from congestion score or cluster label.
    """
    # Center map
    lat0 = df[lat_col].mean()
    lon0 = df[lon_col].mean()
    m = folium.Map(location=[lat0, lon0], zoom_start=6, control_scale=True)

    marker_cluster = MarkerCluster().add_to(m)

    def color_by_score(score):
        # Simple thresholding - returns a hex
        try:
            if np.isnan(score):
                return "#808080"  # grey
            if score >= 0.75:
                return "#800026"
            if score >= 0.5:
                return "#BD0026"
            if score >= 0.25:
                return "#FECC5C"
            return "#FFFFB2"
        except Exception:
            return "#808080"

    for _, row in df.iterrows():
        lat = row.get(lat_col)
        lon = row.get(lon_col)
        if pd.isna(lat) or pd.isna(lon):
            continue
        label = row.get(label_col, "NA")
        congestion = row.get(congestion_col, np.nan)
        popup_text = f"Label: {label}<br>Congestion: {congestion:.3f}" if (congestion is not None and not pd.isna(congestion)) else f"Label: {label}"
        if popup_cols:
            for c in popup_cols:
                popup_text += f"<br>{c}: {row.get(c,'NA')}"
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=color_by_score(congestion),
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(marker_cluster)

    m.save(out_html)
    print(f"Saved folium map to {out_html}")
    return m
