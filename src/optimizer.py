"""
Full optimization pipeline orchestrator.

Routing solver selection
------------------------
1. Gurobi MIP (src/gurobi_model.py) – optimal assignment + sequencing.
   Used when gurobipy is installed and licensed.
2. Nearest-neighbour heuristic (src/routing.py) – fast fallback.
   Used automatically when Gurobi is unavailable or a solve fails.

Usage
-----
    from src.optimizer import run_optimization
    trucks, groups, df_geo = run_optimization(df)
"""

from __future__ import annotations

import pandas as pd

from .clustering import cluster_orders, cluster_summary
from .consolidation import consolidate_orders
from .geolocation import geocode_orders
from .gurobi_model import GUROBI_AVAILABLE, solve_vrp_group
from .routing import route_consolidation_group


def run_optimization(
    df: pd.DataFrame,
    cluster_method: str = "kmeans",
    n_clusters: int = 4,
    eps_km: float = 600.0,
) -> tuple[list[dict], list, pd.DataFrame]:
    """
    Execute the freight optimization pipeline.

    Steps
    -----
    1. Geocode pickup and drop cities.
    2. Cluster orders spatially by drop-city coordinates.
    3. Consolidate into 7-day windows and classify (LTL / TL / ...).
    4. Route each consolidation group:
         • Gurobi MIP VRP (optimal) when gurobipy is available.
         • Nearest-neighbour heuristic (fallback) otherwise.

    Returns
    -------
    trucks   : list of truck assignment dicts (ready to print/export)
    groups   : list of ConsolidationGroup objects
    df_geo   : enriched DataFrame with lat/lng + cluster columns
    """
    if GUROBI_AVAILABLE:
        solver_label = "Gurobi MIP (MTZ-VRP)"
    else:
        solver_label = "nearest-neighbour heuristic (Gurobi not installed)"

    print("  [1/4] Geocoding pickup and drop locations...")
    df_geo = geocode_orders(df)

    print(f"  [2/4] Clustering orders ({cluster_method})...")
    df_geo = cluster_orders(
        df_geo, method=cluster_method, n_clusters=n_clusters, eps_km=eps_km
    )
    summary = cluster_summary(df_geo)
    print(f"        {df_geo['spatial_cluster'].nunique()} clusters identified.")
    for _, row in summary.iterrows():
        print(
            f"        Cluster {row['cluster']:>2}: {row['order_count']} orders  "
            f"-> {row['drop_cities']}"
        )

    print("  [3/4] Consolidating orders (7-day window + freight rules)...")
    groups = consolidate_orders(df_geo)
    print(f"        {len(groups)} consolidation group(s) formed.")

    print(f"  [4/4] Routing trucks ({solver_label}, max {4} stops)...")
    all_trucks: list[dict] = []

    for group in groups:
        if not group.orders:
            continue

        pickup_lat = group.orders[0]["pickup_lat"]
        pickup_lng = group.orders[0]["pickup_lng"]

        # Try Gurobi MIP first; fall back to heuristic if it returns None.
        trucks = solve_vrp_group(group, pickup_lat, pickup_lng)
        if trucks is None:
            trucks = route_consolidation_group(group, pickup_lat, pickup_lng)
            for t in trucks:
                t.setdefault("solver", "Heuristic")
                t.setdefault("solve_info", "nearest-neighbour + greedy bin-pack")

        for truck in trucks:
            truck["group_id"]    = group.group_id
            truck["cluster"]     = group.cluster
            truck["window_start"] = group.window_start
            truck["window_end"]  = group.window_end

        all_trucks.extend(trucks)

    print(f"        {len(all_trucks)} truck(s) dispatched.\n")
    return all_trucks, groups, df_geo
