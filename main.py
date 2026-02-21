"""
Freight Route Optimization – CLI entry point.

Usage
-----
    python main.py [orders.csv] [kmeans|dbscan]

Defaults to  data/sample_orders.csv  with KMeans clustering (4 clusters).
"""

from __future__ import annotations


import sys
import pandas as pd
from src.gurobi_model import GUROBI_AVAILABLE


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
_LINE = "=" * 80
_DASH = "-" * 80


def _section(title: str) -> None:
    print(f"\n{_LINE}")
    print(f"  {title}")
    print(_LINE)


def _print_summary(trucks: list[dict], groups: list, df: pd.DataFrame) -> None:
    _section("OPTIMIZATION SUMMARY")

    print(f"\n  Orders loaded          : {len(df)}")
    print(f"  Spatial clusters       : {df['spatial_cluster'].nunique()}")
    print(f"  Consolidation groups   : {len(groups)}")
    print(f"  Trucks dispatched      : {len(trucks)}")

    # Weight conversion summary
    if "weight_lbs" in df.columns:
        kg_orders = df[df["unit"].str.upper().isin(["KG", "KGS", "KGM"])]
        if not kg_orders.empty:
            print(
                f"\n  KG->LB conversions      : {len(kg_orders)} orders "
                f"({kg_orders['quantity'].sum():,.0f} KG -> "
                f"{kg_orders['weight_lbs'].sum():,.0f} LB)"
            )

    # Shipment-type breakdown
    type_counts: dict[str, int] = {}
    for t in trucks:
        type_counts[t["shipment_type"]] = type_counts.get(t["shipment_type"], 0) + 1

    print("\n  Shipment type breakdown:")
    for stype in ["Individual", "LTL", "TL"]:
        count = type_counts.get(stype, 0)
        bar = "#" * count
        print(f"    {stype:<12} : {count:>2}  {bar}")

    # Group-level overview table
    _section("CONSOLIDATION GROUPS")
    header = (
        f"  {'Group':<10} {'Cluster':>7} {'Window':>22} "
        f"{'Orders':>6} {'Weight (lb)':>12} {'Type':<12}"
    )
    print(header)
    print("  " + "-" * 76)
    for g in groups:
        print(
            f"  {g.group_id:<10} {g.cluster:>7} "
            f"  {g.window_start} -> {g.window_end}  "
            f"{len(g.orders):>6} {g.total_weight_lbs:>12,.1f} {g.shipment_type:<12}"
        )


def _print_truck_detail(trucks: list[dict]) -> None:
    _section("TRUCK ASSIGNMENTS & DELIVERY ROUTES")

    for truck in trucks:
        solver_info = truck.get("solve_info", "")
        solver_tag = f"{truck.get('solver', 'Heuristic')}  [{solver_info}]"
        print(f"\n  Truck ID      : {truck['truck_id']}")
        print(f"  Shipment Type : {truck['shipment_type']}")
        print(f"  Solver        : {solver_tag}")
        print(f"  Cluster       : {truck['cluster']}")
        print(f"  Date Window   : {truck['window_start']} -> {truck['window_end']}")
        print(f"  Total Weight  : {truck['total_weight_lbs']:,.1f} lbs")
        print(f"  Est. Distance : {truck['total_distance_miles']:,.1f} miles")
        print(f"  Stops         : {truck['n_stops']}")
        print()
        print(
            f"  {'Stop':<6} {'Drop City':<20} {'Sales Doc':<14} {'PO Number':<16} "
            f"{'Material':<14} {'Weight (lb)':>11}"
        )
        print("  " + "-" * 84)
        for i, stop in enumerate(truck["route"], start=1):
            print(
                f"  {i:<6} {stop['drop_city']:<20} {stop['sales_document']:<14} "
                f"{stop['po_number']:<16} {stop['material']:<14} "
                f"{stop['weight_lbs']:>11,.1f}"
            )
        print("  " + _DASH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:

    data_file = (
        sys.argv[1] if len(sys.argv) > 1 else "data/sample_orders_kmeans_preproc.csv"
    )
    solver_label = (
        "Gurobi MIP (MTZ-VRP)" if GUROBI_AVAILABLE else "Heuristic (nearest-neighbour)"
    )
    print(f"  Routing solver         : {solver_label}\n")

    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"ERROR: File not found: {data_file}")
        sys.exit(1)

    print(f"  Loaded {len(df)} orders (preprocessed).\n")

    if not GUROBI_AVAILABLE:
        print("ERROR: Gurobi is not available. Cannot run MIP optimization.")
        sys.exit(1)

    print("  Running MIP optimization (Gurobi)...")
    from src.gurobi_model import solve_vrp_group
    from src.consolidation import ConsolidationGroup

    # Reconstruct groups from the preprocessed CSV
    group_cols = [
        "group_id",
        "cluster",
        "window_start",
        "window_end",
        "total_weight_lbs",
        "shipment_type",
    ]
    groups = []
    for group_id, group_df in df.groupby("group_id"):
        group_info = group_df.iloc[0]
        orders = group_df.to_dict("records")
        group = ConsolidationGroup(
            group_id=group_info["group_id"],
            cluster=int(group_info["cluster"]),
            window_start=group_info["window_start"],
            window_end=group_info["window_end"],
            orders=orders,
            total_weight_lbs=float(group_info["total_weight_lbs"]),
            shipment_type=group_info["shipment_type"],
        )
        groups.append(group)

    all_trucks = []
    for group in groups:
        if not group.orders:
            continue
        pickup_lat = group.orders[0]["pickup_lat"]
        pickup_lng = group.orders[0]["pickup_lng"]
        trucks = solve_vrp_group(group, pickup_lat, pickup_lng)
        if trucks is None:
            print(f"    [Group {group.group_id}] Gurobi failed or infeasible.")
            continue
        for truck in trucks:
            truck["group_id"] = group.group_id
            truck["cluster"] = group.cluster
            truck["window_start"] = group.window_start
            truck["window_end"] = group.window_end
        all_trucks.extend(trucks)

    _print_summary(all_trucks, groups, df)
    _print_truck_detail(all_trucks)

    print(f"\n{_LINE}")
    print("  Done.")
    print(_LINE)


if __name__ == "__main__":
    main()
