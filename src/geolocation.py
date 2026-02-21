"""
Geocoding module: resolves city names to (lat, lng) coordinates.

Resolution order:
  1. In-memory CITY_COORDINATES dict (covers all demo cities, instant)
  2. JSON file cache (data/geocache.json) for previously resolved cities
  3. Nominatim (free OpenStreetMap API, rate-limited ~1 req/sec) for unknowns
"""

import json
import os
import time

# ---------------------------------------------------------------------------
# Hardcoded coordinates for all cities used in the demo dataset.
# Keys must exactly match the city strings in the CSV.
# ---------------------------------------------------------------------------
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    # Pickup / Distribution Centre
    "Kansas City MO": (39.0997, -94.5786),
    # Great-Lakes cluster
    "Milwaukee WI":    (43.0389, -87.9065),
    "Indianapolis IN": (39.7684, -86.1581),
    "Detroit MI":      (42.3314, -83.0458),
    "Cleveland OH":    (41.4993, -81.6944),
    "Columbus OH":     (39.9612, -82.9988),
    # Southeast cluster
    "Nashville TN":    (36.1627, -86.7816),
    "Charlotte NC":    (35.2271, -80.8431),
    "New Orleans LA":  (29.9511, -90.0715),
    "Memphis TN":      (35.1495, -90.0490),
    "Birmingham AL":   (33.5186, -86.8104),
    # Texas / Southwest cluster
    "Houston TX":      (29.7604, -95.3698),
    "Dallas TX":       (32.7767, -96.7970),
    "San Antonio TX":  (29.4241, -98.4936),
    "Oklahoma City OK":(35.4676, -97.5164),
    "Tulsa OK":        (36.1540, -95.9928),
    # Northeast cluster
    "Philadelphia PA": (39.9526, -75.1652),
    "Baltimore MD":    (39.2904, -76.6122),
    "Boston MA":       (42.3601, -71.0589),
    "New York NY":     (40.7128, -74.0060),
    "Providence RI":   (41.8240, -71.4128),
}

_CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "geocache.json")


def _load_file_cache() -> dict:
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_file_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def geocode_city(city_name: str) -> tuple[float, float]:
    """Return (lat, lng) for *city_name*. Raises ValueError if unresolvable."""
    # 1 – in-memory dict
    if city_name in CITY_COORDINATES:
        return CITY_COORDINATES[city_name]

    # 2 – file cache
    cache = _load_file_cache()
    if city_name in cache:
        return tuple(cache[city_name])

    # 3 – Nominatim (live API)
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut

        geolocator = Nominatim(user_agent="freight-optimizer-demo")
        time.sleep(1)  # respect rate limit
        location = geolocator.geocode(city_name)
        if location:
            coords = (location.latitude, location.longitude)
            cache[city_name] = list(coords)
            _save_file_cache(cache)
            return coords
    except Exception:
        pass

    raise ValueError(
        f"Could not geocode '{city_name}'. Add it to CITY_COORDINATES or fix the spelling."
    )


def geocode_orders(df):
    """
    Add pickup_lat, pickup_lng, drop_lat, drop_lng columns to *df*.
    Geocodes each unique city once.
    """
    df = df.copy()

    pickup_coords: dict[str, tuple[float, float]] = {}
    drop_coords: dict[str, tuple[float, float]] = {}

    for city in df["first_pick_city"].unique():
        pickup_coords[city] = geocode_city(city)

    for city in df["drop_city"].unique():
        drop_coords[city] = geocode_city(city)

    df["pickup_lat"] = df["first_pick_city"].map(lambda c: pickup_coords[c][0])
    df["pickup_lng"] = df["first_pick_city"].map(lambda c: pickup_coords[c][1])
    df["drop_lat"]   = df["drop_city"].map(lambda c: drop_coords[c][0])
    df["drop_lng"]   = df["drop_city"].map(lambda c: drop_coords[c][1])

    return df
