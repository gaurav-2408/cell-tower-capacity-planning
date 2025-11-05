import os
import pandas as pd
import numpy as np
from scripts.evaluate_metrics import evaluate_metrics

def compute_coverage_metrics(df, dynamic=True, range_min=500, range_max=2000):
    if "tower_range" not in df.columns or "tower_samples" not in df.columns:
        print("Missing columns. Skipping coverage metrics.")
        return {
            "Under Coverage (%)": None,
            "Over Coverage (%)": None,
            "Coverage Ratio (per km)": None,
            "Coverage Ratio Index (0-1)": None
        }

    df_valid = df.dropna(subset=["tower_range", "tower_samples"])
    if df_valid.empty:
        return {
            "Under Coverage (%)": None,
            "Over Coverage (%)": None,
            "Coverage Ratio (per km)": None,
            "Coverage Ratio Index (0-1)": None
        }

    if dynamic:
        range_min = df_valid["tower_range"].quantile(0.25)
        range_max = df_valid["tower_range"].quantile(0.75)
        print(f"[Auto Thresholds] Under < {range_min:.2f} | Over > {range_max:.2f}")

    if df_valid["tower_range"].mean() < 50:
        print("Detected tower_range in kilometers, converting to meters.")
        df_valid["tower_range"] = df_valid["tower_range"] * 1000

    under_coverage = (df_valid["tower_range"] < range_min).mean() * 100
    over_coverage = (df_valid["tower_range"] > range_max).mean() * 100

    total_samples = df_valid["tower_samples"].sum()
    total_range_km = df_valid["tower_range"].sum() / 1000
    coverage_ratio_km = 0 if total_range_km == 0 else total_samples / total_range_km

    eps = 1e-6
    dens = df_valid["tower_samples"] / np.maximum(df_valid["tower_range"], eps)
    p10, p90 = np.nanpercentile(dens, [10, 90])
    if p90 - p10 < eps:
        coverage_ratio_index = 0.0
    else:
        idx = np.clip((dens - p10) / (p90 - p10), 0, 1)
        coverage_ratio_index = float(np.nanmean(idx))

    return {
        "Under Coverage (%)": round(under_coverage, 4),
        "Over Coverage (%)": round(over_coverage, 4),
        "Coverage Ratio (per km)": round(coverage_ratio_km, 4),
        "Coverage Ratio Index (0-1)": round(coverage_ratio_index, 4)
    }

def generate_html_report(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html_sections = []
    html_sections.append("<h1 style='color:#2E86C1;'>Cell Tower Capacity & Clustering Report</h1>")
    html_sections.append("<p>This report summarizes K-Means clustering and coverage efficiency metrics.</p>")

    if "geo_kmeans_label" in df.columns and "servingcell_lat" in df.columns and "servingcell_lon" in df.columns:
        print("Computing K-Means metrics...")
        X_geo = df[["servingcell_lat", "servingcell_lon"]].dropna().values
        labels_geo = df.loc[~df["geo_kmeans_label"].isna(), "geo_kmeans_label"].astype(int).values
        geo_metrics = evaluate_metrics(X_geo, labels_geo, sample_size=5000)
        html_sections.append("<h2 style='color:#1F618D;'>K-Means Clustering Metrics</h2>")
        html_sections.append(pd.DataFrame(geo_metrics.items(), columns=["Metric", "Value"]).to_html(index=False, border=0, justify="center", float_format="%.4f"))

    html_sections.append("<h2 style='color:#117A65;'>Coverage Efficiency Metrics</h2>")
    coverage_metrics = compute_coverage_metrics(df)
    html_sections.append(pd.DataFrame(coverage_metrics.items(), columns=["Metric", "Value"]).to_html(index=False, border=0, justify="center", float_format="%.4f"))

    html_content = """
    <html>
    <head>
    <title>Cell Tower Clustering Report</title>
    <style>
    body {font-family: Arial; margin: 40px; background-color: #f8f9f9; color: #212F3D;}
    h1, h2, h3 {font-family: Segoe UI;}
    table {border-collapse: collapse; margin: 20px 0; width: 70%;}
    th, td {border: 1px solid #ccc; padding: 8px 12px; text-align: left;}
    th {background-color: #3498DB; color: white;}
    tr:nth-child(even) {background-color: #f2f2f2;}
    .footer {margin-top: 40px; font-size: 0.9em; color: gray;}
    </style>
    </head>
    <body>
    """ + "\n".join(html_sections) + """
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML report generated successfully at: {output_path}")
