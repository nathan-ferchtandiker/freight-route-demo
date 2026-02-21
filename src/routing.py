"""
Multi-stop routing with a 4-stop constraint.

Algorithm
---------
1. If a consolidation group has > MAX_STOPS orders OR total weight > TL_MAX_LBS,
   split it into sub-trucks using a greedy bin-packer (weight + stop limits).
2. For each sub-truck, sequence deliveries with the nearest-neighbour heuristic
   starting from the pickup (DC) location.
3. Re-classify each truck's shipment type based on its own weight.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .consolidation import ConsolidationGroup, LTL_MAX_LBS, TL_MAX_LBS
from .distance import haversine_miles

MAX_STOPS = 4


# ---------------------------------------------------------------------------
# Nearest-neighbour TSP
# ---------------------------------------------------------------------------
def _nearest_neighbor_route(
    pickup_lat: float,
    pickup_lng: float,
    orders: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """
    Greedy nearest-neighbour route starting from the pickup location.

    Returns
    -------
    (ordered_stops, total_distance_miles)
    """
    if not orders:
        return [], 0.0

    unvisited = list(orders)
    route: list[dict[str, Any]] = []
    cur_lat, cur_lng = pickup_lat, pickup_lng
    total_dist = 0.0

    while unvisited:
        dists = [
            haversine_miles(cur_lat, cur_lng, o["drop_lat"], o["drop_lng"])
            for o in unvisited
        ]
        nearest_idx = int(np.argmin(dists))
        nearest = unvisited.pop(nearest_idx)
        total_dist += dists[nearest_idx]
        route.append(nearest)
        cur_lat = nearest["drop_lat"]
        cur_lng = nearest["drop_lng"]

    return route, round(total_dist, 1)


# ---------------------------------------------------------------------------
# Weight-aware bin-packer
# ---------------------------------------------------------------------------
def _split_into_trucks(
    orders: list[dict[str, Any]],
    max_stops: int = MAX_STOPS,
    max_weight: float = TL_MAX_LBS,
) -> list[list[dict[str, Any]]]:
    """
    Partition *orders* into trucks such that each truck has at most
    *max_stops* stops and total weight ≤ *max_weight*.

    Uses a simple greedy first-fit algorithm.
    """
    trucks: list[list[dict[str, Any]]] = []
    remaining = list(orders)

    while remaining:
        truck: list[dict[str, Any]] = []
        truck_weight = 0.0

        for order in list(remaining):
            if len(truck) >= max_stops:
                break
            w = order.get("weight_lbs", 0.0)
            if truck_weight + w <= max_weight or not truck:
                truck.append(order)
                truck_weight += w
                remaining.remove(order)

        trucks.append(truck)

    return trucks


# ---------------------------------------------------------------------------
# Classify a single truck's freight type
# ---------------------------------------------------------------------------
def _classify_truck(weight_lbs: float, n_stops: int) -> str:
    if n_stops == 1:
        return "Individual"
    if weight_lbs < LTL_MAX_LBS:
        return "LTL"
    return "TL"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def route_consolidation_group(
    group: ConsolidationGroup,
    pickup_lat: float,
    pickup_lng: float,
) -> list[dict[str, Any]]:
    """
    Route all orders in *group*, splitting into multiple trucks when needed.

    Returns a list of truck dicts, each containing:
        truck_id, shipment_type, orders, total_weight_lbs,
        route (delivery-sequenced orders), total_distance_miles,
        n_stops
    """
    orders = group.orders

    # Determine whether splitting is required
    needs_split = (
        len(orders) > MAX_STOPS
        or group.total_weight_lbs >= TL_MAX_LBS
    )

    if needs_split:
        truck_order_groups = _split_into_trucks(orders)
    else:
        truck_order_groups = [orders]

    trucks: list[dict[str, Any]] = []
    for i, truck_orders in enumerate(truck_order_groups, start=1):
        route, dist = _nearest_neighbor_route(pickup_lat, pickup_lng, truck_orders)
        truck_weight = round(sum(o.get("weight_lbs", 0.0) for o in truck_orders), 1)
        stype = _classify_truck(truck_weight, len(truck_orders))

        trucks.append(
            {
                "truck_id": f"{group.group_id}-T{i}",
                "shipment_type": stype,
                "orders": truck_orders,
                "total_weight_lbs": truck_weight,
                "route": route,
                "total_distance_miles": dist,
                "n_stops": len(route),
            }
        )

    return trucks
