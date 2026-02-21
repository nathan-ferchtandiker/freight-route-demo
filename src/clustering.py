"""
Spatial clustering of orders by drop (delivery) city coordinates.

Supported methods
-----------------
kmeans  – KMeans with a fixed or auto-detected number of clusters.
          Deterministic (random_state=42), fast, good for well-separated regions.
dbscan  – DBSCAN with haversine metric.
          No need to specify k; discovers clusters and marks outliers as -1
          (treated as individual shipments).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score


def cluster_orders(
    df: pd.DataFrame,
    method: str = "kmeans",
    n_clusters: int | None = 4,
    eps_km: float = 600.0,
) -> pd.DataFrame:
    """
    Cluster orders by drop-city coordinates.

    Parameters
    ----------
    df         : DataFrame with drop_lat / drop_lng columns.
    method     : 'kmeans' or 'dbscan'.
    n_clusters : Number of clusters for KMeans (None → auto via silhouette).
    eps_km     : Epsilon radius in km for DBSCAN.

    Returns
    -------
    df with a new 'spatial_cluster' integer column.
    """
    df = df.copy()
    coords = df[["drop_lat", "drop_lng"]].values

    if method == "kmeans":
        k = n_clusters if n_clusters is not None else _optimal_k(coords)
        k = max(1, min(k, len(df)))
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        df["spatial_cluster"] = model.fit_predict(coords)

    elif method == "dbscan":
        coords_rad = np.radians(coords)
        eps_rad = eps_km / 6371.0
        model = DBSCAN(
            eps=eps_rad,
            min_samples=1,
            algorithm="ball_tree",
            metric="haversine",
        )
        df["spatial_cluster"] = model.fit_predict(coords_rad)
        # DBSCAN labels noise as -1; remap each noise point to its own cluster ID
        df = _remap_noise(df)

    else:
        raise ValueError(f"Unknown clustering method '{method}'. Choose 'kmeans' or 'dbscan'.")

    return df


def _optimal_k(coords: np.ndarray, max_k: int = 8) -> int:
    """Pick k that maximises the silhouette score (2 ≤ k ≤ max_k)."""
    if len(coords) < 4:
        return min(2, len(coords))

    best_k, best_score = 2, -1.0
    for k in range(2, min(max_k + 1, len(coords))):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(coords)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(coords, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def _remap_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    DBSCAN marks outliers as -1.  Assign each outlier its own unique cluster ID
    so downstream logic treats it as a single-order cluster.
    """
    next_id = df["spatial_cluster"].max() + 1
    for idx in df.index[df["spatial_cluster"] == -1]:
        df.at[idx, "spatial_cluster"] = next_id
        next_id += 1
    return df


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame: cluster → cities, order count, centre."""
    rows = []
    for cid in sorted(df["spatial_cluster"].unique()):
        sub = df[df["spatial_cluster"] == cid]
        rows.append(
            {
                "cluster": cid,
                "order_count": len(sub),
                "drop_cities": ", ".join(sorted(sub["drop_city"].unique())),
                "centre_lat": round(sub["drop_lat"].mean(), 4),
                "centre_lng": round(sub["drop_lng"].mean(), 4),
            }
        )
    return pd.DataFrame(rows)
