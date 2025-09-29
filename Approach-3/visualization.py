import matplotlib.pyplot as plt
import numpy as np
import folium
from folium.plugins import MarkerCluster

def plot_hexbin(df, feature="throughput", save_path="hexbin.png"):
    plt.figure(figsize=(10, 6))
    hb = plt.hexbin(df["longitude"], df["latitude"],
                    C=df[feature], gridsize=60,
                    cmap="YlOrRd", reduce_C_function=np.mean)
    plt.colorbar(hb, label=f"Avg {feature}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Hexbin of {feature}")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved hexbin map: {save_path}")


def plot_folium_clusters(df, label_col="geo_kmeans_label", lat_col="tower_lat", lon_col="tower_lon",
                         out_html="map.html", feature="throughput"):
    lat0, lon0 = df[lat_col].mean(), df[lon_col].mean()
    m = folium.Map(location=[lat0, lon0], zoom_start=7)

    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df.dropna(subset=[lat_col, lon_col]).iterrows():
        popup = f"{label_col}: {row.get(label_col, 'NA')}<br>{feature}: {row.get(feature, 'NA')}"
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4,
            color="blue",
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(marker_cluster)

    m.save(out_html)
    print(f"Saved interactive map: {out_html}")
