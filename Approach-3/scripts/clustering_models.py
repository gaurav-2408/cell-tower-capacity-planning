from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

def cluster_geo_kmeans(df, n_clusters=5, lat_col=None, lon_col=None):
    if lat_col is None or lon_col is None:
        if "servingcell_lat" in df.columns and "servingcell_lon" in df.columns:
            lat_col, lon_col = "servingcell_lat", "servingcell_lon"
            print(f"Using serving cell coordinates: {lat_col}, {lon_col}")
        elif "latitude" in df.columns and "longitude" in df.columns:
            lat_col, lon_col = "latitude", "longitude"
            print(f"Using measurement coordinates: {lat_col}, {lon_col}")
        else:
            raise KeyError("No suitable latitude/longitude columns found in dataset")

    X = df[[lat_col, lon_col]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    df.loc[X.index, "geo_kmeans_label"] = labels
    return df, kmeans


def cluster_geo_dbscan(df, eps=0.06, min_samples=10, lat_col=None, lon_col=None, sample_size=10000):
    if lat_col is None or lon_col is None:
        if "servingcell_lat" in df.columns and "servingcell_lon" in df.columns:
            lat_col, lon_col = "servingcell_lat", "servingcell_lon"
            print(f"Using serving cell coordinates: {lat_col}, {lon_col}")
        elif "latitude" in df.columns and "longitude" in df.columns:
            lat_col, lon_col = "latitude", "longitude"
            print(f"Using measurement coordinates: {lat_col}, {lon_col}")
        else:
            raise KeyError("No suitable latitude/longitude columns found in dataset")

    X = df[[lat_col, lon_col]].dropna()
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    df.loc[X.index, "geo_dbscan_label"] = labels
    return df, db


def cluster_features_kmeans(df, feature_list, n_clusters=5):
    X = df[feature_list].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    df.loc[X.index, "feature_kmeans_label"] = labels
    return df, kmeans
