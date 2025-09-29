"""
clustering.py
Implements KMeans and DBSCAN clustering pipelines and saves cluster labels.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

def run_kmeans(df, X, n_clusters=4, random_state=42, save_model_path=None):
    """
    Fit KMeans on scaled features X (DataFrame) and attach label column to df.
    Returns df with "kmeans_label" and optionally saves model with joblib.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optionally run PCA for faster/sanity visualization (not required)
    # pca = PCA(n_components=min(10, X_scaled.shape[1]))
    # X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    out = df.copy()
    out["kmeans_label"] = labels
    # Add cluster centroids in feature space (inverse scaled)
    out.attrs["kmeans_model"] = kmeans
    out.attrs["kmeans_scaler"] = scaler

    if save_model_path:
        joblib.dump({"model": kmeans, "scaler": scaler}, save_model_path)

    return out

def run_dbscan(df, X, eps=0.5, min_samples=5):
    """
    Fit DBSCAN on scaled features X (DataFrame) and attach label column to df.
    DBSCAN label -1 indicates noise.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(X_scaled)

    out = df.copy()
    out["dbscan_label"] = labels
    out.attrs["dbscan_model"] = db
    out.attrs["dbscan_scaler"] = scaler
    return out

def reduce_dimensionality_for_plot(X, scaler=None, n_components=2):
    """
    Optional helper to get 2D coords for plotting clusters (PCA).
    """
    from sklearn.decomposition import PCA
    if scaler:
        Xs = scaler.transform(X)
    else:
        Xs = X.values
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(Xs)
    return coords
