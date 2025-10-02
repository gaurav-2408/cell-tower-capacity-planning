import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import resample
from scipy.spatial.distance import cdist


def dunn_index(X, labels):

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0

    intra_dists = []
    for lbl in unique_labels:
        cluster_points = X[labels == lbl]
        if len(cluster_points) > 1:
            dists = cdist(cluster_points, cluster_points, metric="euclidean")
            intra_dists.append(np.max(dists))
        else:
            intra_dists.append(0)
    max_intra = max(intra_dists)


    inter_dists = []
    for i, lbl1 in enumerate(unique_labels):
        for lbl2 in unique_labels[i+1:]:
            c1 = X[labels == lbl1]
            c2 = X[labels == lbl2]
            dists = cdist(c1, c2, metric="euclidean")
            inter_dists.append(np.min(dists))
    min_inter = min(inter_dists)

    return min_inter / max_intra if max_intra > 0 else 0


def evaluate_metrics(X, labels, sample_size=5000):

    results = {}

    if len(set(labels)) > 1:  # Metrics need >= 2 clusters
        if len(X) > sample_size:
            X_s, labels_s = resample(X, labels, n_samples=sample_size, random_state=42)
            results["Silhouette Score (sample)"] = silhouette_score(X_s, labels_s)
        else:
            results["Silhouette Score"] = silhouette_score(X, labels)

        results["Davies-Bouldin Index"] = davies_bouldin_score(X, labels)
        results["Calinski-Harabasz Index"] = calinski_harabasz_score(X, labels)

        if len(X) > sample_size:
            X_s, labels_s = resample(X, labels, n_samples=sample_size, random_state=42)
            results["Dunn Index (sample)"] = dunn_index(X_s, labels_s)
        else:
            results["Dunn Index"] = dunn_index(X, labels)
    else:
        results = {m: None for m in [
            "Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index", "Dunn Index"
        ]}

    return results
