"""Haversine distance utilities."""

import numpy as np

_EARTH_RADIUS_MILES = 3958.8


def haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return great-circle distance in miles between two (lat, lng) points."""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return _EARTH_RADIUS_MILES * 2 * np.arcsin(np.sqrt(a))


def pairwise_distance_matrix(coords: list[tuple[float, float]]) -> np.ndarray:
    """
    Return an (n x n) matrix of haversine distances in miles.
    *coords* is a list of (lat, lng) tuples in degrees.
    """
    n = len(coords)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i, j] = haversine_miles(
                    coords[i][0], coords[i][1], coords[j][0], coords[j][1]
                )
    return mat
