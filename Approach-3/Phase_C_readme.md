📖 Cell Tower Capacity Planning – Phase C (Clustering & GeoSpatial Analysis)
🔎 Overview

This phase extends our cell tower capacity planning project into clustering and geospatial analysis.

We use unsupervised learning (K-Means, DBSCAN) combined with geographic visualization (hexbin maps, Folium interactive maps) to:

Group towers based on traffic load, congestion, utilization, and signal quality

Identify hotspots (overloaded towers needing upgrades) and blind spots (poor coverage areas)

Provide decision support for new tower placement, capacity upgrades, and optimization

📂 Project Structure
cell_tower_phase_c/
│
├── data/
│   └── towers.csv              # Input dataset (preprocessed)
│
├── scripts/
│   ├── utils.py                # Data loading, derived features
│   ├── clustering.py           # KMeans, DBSCAN
│   ├── evaluate.py             # Cluster summaries, silhouette score
│   ├── visualize.py            # Hexbin plots + Folium maps
│
├── outputs/                    # Auto-generated results (CSV, PNG, HTML maps)
│
├── requirements.txt            # Dependencies
└── main.py                     # Pipeline orchestration

📊 Input Data

Your dataset (towers.csv) should include attributes across traffic load, congestion, coverage, geospatial, and signal quality.
Examples:

Traffic Load Indicators: dl_bitrate, ul_bitrate, samples, throughput_degradation

Congestion Features: traffic_density, load_factor, congestion_score

Coverage Features: range, coverage_efficiency, resource_utilization

Geo Attributes: latitude, longitude, cell_density, distance

Signal Quality: rsrp, rsrq, snr, cqi

Performance: throughput, peak_load_ratio, user_density

Identifiers: cell_id, mcc, mnc, pci

If some fields are missing, the code fills them with defaults (NaN → handled during preprocessing).

🛠️ Installation

Clone repository / copy project folder

Create virtual environment

python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows


Install requirements

pip install -r requirements.txt

🚀 Running the Pipeline

Run the Phase C pipeline with:

python main.py


This will:

Load and preprocess data

Compute derived features (e.g., traffic_density, congestion_score)

Cluster towers with KMeans and DBSCAN

Evaluate with silhouette score & generate cluster summaries

Save results in /outputs/

📂 Outputs

Clustered Datasets

clusters_kmeans.csv

clusters_dbscan.csv

Cluster Summaries

kmeans_summary.csv

dbscan_summary.csv

Visualizations

hexbin_congestion.png (congestion heatmap)

map_kmeans.html (interactive Folium map with KMeans clusters)

map_dbscan.html (interactive Folium map with DBSCAN clusters)

📈 Algorithms Used
🔹 K-Means

Groups towers into k clusters (configurable)

Best for globally balanced patterns

Evaluated using silhouette score

🔹 DBSCAN

Density-based clustering

Detects dense hotspots and labels outliers (-1)

Ideal for irregular geographic distributions

🔹 Hexbin Mapping

Aggregates towers into hexagonal bins

Highlights average congestion score per region

Great for coverage planning visualization

📊 Example Insights

From clustering + maps, we can derive:

High-congestion clusters → Towers likely to need upgrades

Blind spots (low signal_quality_ranking, high distance) → Candidate sites for new towers

Underutilized clusters → Towers that may be redundant / resources reallocated

Geo hotspot overlap with anomalies (Phase B) → Validate anomaly causes

⚙️ Configuration

Edit main.py to adjust:

Number of clusters (KMeans) → n_clusters=5

DBSCAN params → eps=0.5, min_samples=5

Features used for clustering → Modify prepare_features_for_clustering() in utils.py

📌 Next Steps

Add H3-based hexagon aggregation for scalable geospatial analysis

Integrate into AWS Step Functions pipeline for automation

Explore Reinforcement Learning for dynamic capacity management