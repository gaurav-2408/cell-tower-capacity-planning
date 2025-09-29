"""
evaluate.py
Simple clustering diagnostics: silhouette score and cluster summaries.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

def silhouette_for_labels(X, labels):
    """
    Compute silhouette score only if >1 cluster and not all single label.
    """
    try:
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels and len(unique_labels)==1):
            return np.nan
        return silhouette_score(X, labels)
    except Exception:
        return np.nan

def cluster_summary(df, label_col, features):
    """
    Return a short summary DataFrame grouped by label_col with mean/std/count of features.
    """
    summary = df.groupby(label_col)[features].agg(["mean","std","count"])
    return summary
