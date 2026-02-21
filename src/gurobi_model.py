"""
Gurobi-based multi-truck Vehicle Routing Problem (VRP) solver.

Mathematical Formulation
------------------------
Given a consolidation group of n orders and a depot (pickup city), find the
minimum number of trucks and the minimum-distance delivery sequence such that:

  • Every order is delivered by exactly one truck.
  • Each truck visits at most MAX_STOPS = 4 delivery locations.
  • Each truck's total load does not exceed TL_MAX_LBS = 45,000 lb.

Decision variables
------------------
  x[i, j, k] ∈ {0,1}  — truck k travels arc (i -> j).
                          Nodes: 0 = depot, 1..n = delivery stops.
                          Includes zero-cost return arcs (j = 0) so MTZ
                          subtour elimination applies to a closed tour.
  y[i, k]    ∈ {0,1}  — order i is served by truck k.
  z[k]       ∈ {0,1}  — truck k is activated (used).
  u[i, k]    ≥ 0 cont — position of node i in truck k's route (MTZ).

Objective (lexicographic via big-M weighting)
----------------------------------------------
  min  BIG_M · Σ_k z[k]  +  Σ_{k,i≠j, j≠0} dist[i][j] · x[i,j,k]

  The BIG_M term (100 000 per truck) always dominates the distance term
  (max route ≈ 10 000 miles), so the solver first minimises the truck count
  and then minimises total one-way delivery distance within that count.

Constraints
-----------
  C1.  Σ_k y[i,k] = 1                  ∀ delivery stop i      (each order served once)
  C2.  Σ_j x[0,j,k] = z[k]             ∀ k                    (truck departs depot iff used)
  C3.  Σ_i x[i,0,k] = z[k]             ∀ k                    (zero-cost return to depot)
  C4.  Σ_j x[j,i,k] = y[i,k]           ∀ i ∈ stops, k         (flow in = assignment)
  C5.  Σ_j x[i,j,k] = y[i,k]           ∀ i ∈ stops, k         (flow out = assignment)
  C6.  Σ_i y[i,k] ≤ MAX_STOPS          ∀ k                    (stop limit)
  C7.  Σ_i w[i]·y[i,k] ≤ TL_MAX_LBS   ∀ k                    (weight limit)
  C8.  u[i,k] - u[j,k] + n·x[i,j,k] ≤ n-1   ∀ i,j ∈ stops, i≠j, k   (MTZ)
  C9.  u[0,k] = 0                       ∀ k                    (depot anchor)
  C10. z[k] ≥ z[k+1]                   ∀ k                    (symmetry breaking)

Availability
------------
Requires the `gurobipy` package and a Gurobi licence.  The free size-limited
licence (shipped with `pip install gurobipy`) supports up to 2 000 variables
and 2 000 constraints — sufficient for groups of up to ~12 orders.
When Gurobi is unavailable or the solve fails, `solve_vrp_group` returns None
and the caller falls back to the nearest-neighbour heuristic.
"""

from __future__ import annotations

from typing import Any

from .consolidation import ConsolidationGroup, LTL_MAX_LBS, TL_MAX_LBS
from .distance import haversine_miles

# ---------------------------------------------------------------------------
# Gurobi import (optional dependency)
# ---------------------------------------------------------------------------
try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

MAX_STOPS: int = 4
_BIG_M: float = 100_000.0  # per-truck penalty in the objective


# ===========================================================================
# Public API
# ===========================================================================


def solve_vrp_group(
    group: ConsolidationGroup,
    pickup_lat: float,
    pickup_lng: float,
) -> list[dict[str, Any]] | None:
    """
    Solve the multi-truck open-path VRP for *group* using Gurobi.

    Returns
    -------
    List of truck dicts (same schema as `routing.route_consolidation_group`),
    or **None** if Gurobi is unavailable or the solve fails.
    """
    if not GUROBI_AVAILABLE:
        return None

    orders = group.orders
    n = len(orders)

    if n == 0:
        return []

    # ------------------------------------------------------------------
    # Node layout: 0 = depot, 1..n = delivery stops
    # ------------------------------------------------------------------
    lats = [pickup_lat] + [o["drop_lat"] for o in orders]
    lngs = [pickup_lng] + [o["drop_lng"] for o in orders]
    weights = [0.0] + [o.get("weight_lbs", 0.0) for o in orders]

    N = n + 1
    stops = list(range(1, N))
    nodes = list(range(N))

    # Pairwise distances (miles); return arcs (j=0) cost 0 in objective.
    dist_m = [
        [
            haversine_miles(lats[i], lngs[i], lats[j], lngs[j]) if i != j else 0.0
            for j in range(N)
        ]
        for i in range(N)
    ]

    K_max = n  # upper bound: worst case every order on its own truck
    K = list(range(1, K_max + 1))

    # ------------------------------------------------------------------
    # Build and solve the MIP
    # ------------------------------------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)  # suppress console output
    env.setParam("LogFile", "")
    env.start()

    m = gp.Model(env=env)
    m.setParam("TimeLimit", 120)
    m.setParam("MIPGap", 0.005)  # stop at 0.5 % optimality gap
    m.setParam("Threads", 4)

    # ---- Variables ------------------------------------------------
    arc_keys = [(i, j, k) for k in K for i in nodes for j in nodes if i != j]
    x = m.addVars(arc_keys, vtype=GRB.BINARY, name="x")
    y = m.addVars([(i, k) for i in stops for k in K], vtype=GRB.BINARY, name="y")
    z = m.addVars(K, vtype=GRB.BINARY, name="z")
    u = m.addVars(
        [(i, k) for i in nodes for k in K], lb=0, ub=n, vtype=GRB.CONTINUOUS, name="u"
    )

    # Set branching priorities: structural decisions first
    # Truck activation (highest priority)
    for k in K:
        z[k].BranchPriority = 10
    
    # Order assignments (medium priority)  
    for i in stops:
        for k in K:
            y[i, k].BranchPriority = 5
    
    # Arc routing decisions (lowest priority)
    for (i, j, k) in arc_keys:
        x[i, j, k].BranchPriority = 1

    # ---- Objective ------------------------------------------------
    # One-way delivery cost only (return arcs to node 0 have zero cost).
    obj = _BIG_M * gp.quicksum(z[k] for k in K) + gp.quicksum(
        (dist_m[i][j] if j != 0 else 0.0) * x[i, j, k] for (i, j, k) in arc_keys
    )
    m.setObjective(obj, GRB.MINIMIZE)

    # ---- Constraints ----------------------------------------------
    # C1: every order served exactly once
    for i in stops:
        m.addConstr(gp.quicksum(y[i, k] for k in K) == 1, f"serve_{i}")

    # Symmetry breaking: assign first order to first truck
    m.addConstr(y[1, 1] == 1, "sym_break_first_order")

    for k in K:
        # C2: truck departs depot iff activated
        m.addConstr(gp.quicksum(x[0, j, k] for j in stops) == z[k], f"dep_out_{k}")
        # C3: truck returns to depot iff activated (zero-cost arc)
        m.addConstr(gp.quicksum(x[i, 0, k] for i in stops) == z[k], f"dep_ret_{k}")
        # C6: stop limit
        m.addConstr(gp.quicksum(y[i, k] for i in stops) <= MAX_STOPS, f"maxstops_{k}")
        # C7: weight limit
        m.addConstr(
            gp.quicksum(weights[i] * y[i, k] for i in stops) <= TL_MAX_LBS,
            f"weight_{k}",
        )
        # Only deliver on active trucks
        for i in stops:
            m.addConstr(y[i, k] <= z[k], f"active_truck_{i}_{k}")
        # C9: depot position anchor (MTZ)
        m.addConstr(u[0, k] == 0, f"depot_pos_{k}")

    # C4 & C5: flow conservation at delivery stops
    for i in stops:
        for k in K:
            m.addConstr(
                gp.quicksum(x[j, i, k] for j in nodes if j != i) == y[i, k],
                f"arrive_{i}_{k}",
            )
            m.addConstr(
                gp.quicksum(x[i, j, k] for j in nodes if j != i) == y[i, k],
                f"depart_{i}_{k}",
            )

    # C8: MTZ subtour elimination (delivery stops only)
    for i in stops:
        for j in stops:
            if i != j:
                for k in K:
                    m.addConstr(
                        u[i, k] - u[j, k] + n * x[i, j, k] <= n - 1,
                        f"mtz_{i}_{j}_{k}",
                    )

    # C10: symmetry breaking — use lower-indexed trucks first
    for idx in range(len(K) - 1):
        m.addConstr(z[K[idx]] >= z[K[idx + 1]], f"sym_{idx}")

    # ---- Solve ----------------------------------------------------
    m.optimize()

    if m.SolCount == 0:
        m.dispose()
        env.dispose()
        return None

    status_str = (
        "Optimal"
        if m.Status == GRB.OPTIMAL
        else f"Feasible (gap={m.MIPGap * 100:.2f}%)"
    )

    # ---- Extract trucks from solution -----------------------------
    trucks = _extract_trucks(
        m,
        x,
        y,
        z,
        u,
        K,
        stops,
        nodes,
        orders,
        weights,
        group,
        pickup_lat,
        pickup_lng,
        status_str,
    )

    m.dispose()
    env.dispose()
    return trucks


# ===========================================================================
# Private helpers
# ===========================================================================


def _extract_trucks(
    m,
    x,
    y,
    z,
    u,
    K,
    stops,
    nodes,
    orders,
    weights,
    group,
    pickup_lat,
    pickup_lng,
    status_str,
) -> list[dict[str, Any]]:
    """Read variable values and build the truck-assignment list."""
    trucks: list[dict[str, Any]] = []
    truck_counter = 1

    for k in K:
        if z[k].X < 0.5:
            continue

        # Orders assigned to truck k
        assigned = [i for i in stops if y[i, k].X > 0.5]
        if not assigned:
            continue

        # Reconstruct delivery sequence via MTZ u-values (most robust).
        # Fallback: trace x arcs if u values tie (rare).
        try:
            route_indices = sorted(assigned, key=lambda i: u[i, k].X)
        except Exception:
            route_indices = _trace_arcs(x, k, stops, nodes)
            route_indices = [i for i in route_indices if i in assigned]
            if len(route_indices) != len(assigned):
                route_indices = assigned

        truck_orders = [orders[i - 1] for i in route_indices]
        truck_weight = round(sum(weights[i] for i in assigned), 1)
        n_stops = len(truck_orders)

        # One-way delivery distance (depot -> stop 1 -> ... -> stop m)
        route_dist = 0.0
        prev_lat, prev_lng = pickup_lat, pickup_lng
        for o in truck_orders:
            route_dist += haversine_miles(
                prev_lat, prev_lng, o["drop_lat"], o["drop_lng"]
            )
            prev_lat, prev_lng = o["drop_lat"], o["drop_lng"]

        stype = _classify(truck_weight, n_stops)
        trucks.append(
            _make_truck(
                group.group_id,
                truck_counter,
                truck_orders,
                truck_weight,
                truck_orders,
                round(route_dist, 1),
                stype,
                "Gurobi",
                status_str,
            )
        )
        truck_counter += 1

    return trucks


def _trace_arcs(x, k, stops, nodes) -> list[int]:
    """Follow x[·,·,k] arcs from the depot to recover the delivery sequence."""
    next_nd: dict[int, int] = {}
    all_nd = [0] + list(stops)
    for i in all_nd:
        for j in all_nd:
            if i != j and (i, j, k) in x and x[i, j, k].X > 0.5:
                next_nd[i] = j
                break

    route: list[int] = []
    cur, visited = 0, {0}
    while cur in next_nd:
        nxt = next_nd[cur]
        if nxt == 0 or nxt in visited:
            break
        route.append(nxt)
        visited.add(nxt)
        cur = nxt
    return route


def _classify(weight_lbs: float, n_stops: int) -> str:
    if n_stops == 1:
        return "Individual"
    if weight_lbs < LTL_MAX_LBS:
        return "LTL"
    return "TL"


def _make_truck(
    group_id: str,
    truck_num: int,
    orders: list[dict],
    weight_lbs: float,
    route: list[dict],
    total_dist: float,
    shipment_type: str,
    solver: str,
    solve_info: str,
) -> dict[str, Any]:
    return {
        "truck_id": f"{group_id}-T{truck_num}",
        "shipment_type": shipment_type,
        "orders": orders,
        "total_weight_lbs": weight_lbs,
        "route": route,
        "total_distance_miles": total_dist,
        "n_stops": len(route),
        "solver": solver,
        "solve_info": solve_info,
    }
