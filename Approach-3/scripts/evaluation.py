from sklearn.metrics import silhouette_score
import numpy as np

def compute_silhouette(X, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return None
    return silhouette_score(X, labels)


def cluster_summary(df, label_col, features):
    return df.groupby(label_col)[features].agg(["mean", "std", "count"])
