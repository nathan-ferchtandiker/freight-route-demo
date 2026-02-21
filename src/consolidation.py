"""
Order consolidation logic.

Steps
-----
1. Convert all weights to pounds (KG → LB).
2. Within each spatial cluster, group orders into rolling 7-day windows
   (window opens on the earliest delivery date and closes 7 calendar days later).
3. Classify each window as Individual / LTL / TL / Split-TL based on total weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KG_TO_LB: float = 2.20462
LTL_MAX_LBS: float = 18_000.0   # below this → LTL
TL_MAX_LBS: float = 45_000.0    # below this → TL; at/above → Split-TL


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ConsolidationGroup:
    group_id: str
    cluster: int
    window_start: str
    window_end: str
    orders: list[dict[str, Any]]
    total_weight_lbs: float
    shipment_type: str   # 'Individual' | 'LTL' | 'TL' | 'Split-TL'


# ---------------------------------------------------------------------------
# Weight conversion
# ---------------------------------------------------------------------------
def convert_to_lbs(quantity: float, unit: str) -> float:
    """Convert *quantity* in *unit* to pounds."""
    u = unit.strip().upper()
    if u in ("KG", "KGS", "KGM"):
        return float(quantity) * KG_TO_LB
    if u in ("LB", "LBS", "LBR", "LBM"):
        return float(quantity)
    # Fallback: assume LB for unknown units
    return float(quantity)


def add_weight_lbs(df: pd.DataFrame) -> pd.DataFrame:
    """Append a *weight_lbs* column (converted from quantity + unit)."""
    df = df.copy()
    df["weight_lbs"] = df.apply(
        lambda r: convert_to_lbs(r["quantity"], r["unit"]), axis=1
    )
    return df


# ---------------------------------------------------------------------------
# 7-day rolling window grouping
# ---------------------------------------------------------------------------
def rolling_7day_groups(cluster_df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Partition *cluster_df* (already filtered to one cluster) into groups
    where every order's delivery date falls within 7 calendar days of the
    earliest date in that group.
    """
    cluster_df = cluster_df.sort_values("requested_delivery_date").reset_index(drop=True)

    if cluster_df.empty:
        return []

    groups: list[pd.DataFrame] = []
    window_start = cluster_df.at[0, "requested_delivery_date"]
    current_rows: list[int] = [0]

    for i in range(1, len(cluster_df)):
        date = cluster_df.at[i, "requested_delivery_date"]
        if (date - window_start).days <= 7:
            current_rows.append(i)
        else:
            groups.append(cluster_df.iloc[current_rows].copy())
            window_start = date
            current_rows = [i]

    if current_rows:
        groups.append(cluster_df.iloc[current_rows].copy())

    return groups


# ---------------------------------------------------------------------------
# Freight classification
# ---------------------------------------------------------------------------
def classify_shipment(total_weight_lbs: float, n_orders: int) -> str:
    if n_orders == 1:
        return "Individual"
    if total_weight_lbs < LTL_MAX_LBS:
        return "LTL"
    if total_weight_lbs < TL_MAX_LBS:
        return "TL"
    return "Split-TL"   # needs to be broken across multiple trucks


# ---------------------------------------------------------------------------
# Main consolidation pipeline
# ---------------------------------------------------------------------------
def consolidate_orders(df: pd.DataFrame) -> list[ConsolidationGroup]:
    """
    Run the full consolidation pipeline on a geocoded + clustered DataFrame.

    Returns a list of ConsolidationGroup objects ready for routing.
    """
    df = add_weight_lbs(df)
    df["requested_delivery_date"] = pd.to_datetime(df["requested_delivery_date"])

    groups: list[ConsolidationGroup] = []
    counter = 1

    for cluster_id in sorted(df["spatial_cluster"].unique()):
        cluster_df = df[df["spatial_cluster"] == cluster_id].copy()
        windows = rolling_7day_groups(cluster_df)

        for window_df in windows:
            orders = window_df.to_dict("records")
            total_w = window_df["weight_lbs"].sum()
            n = len(window_df)
            stype = classify_shipment(total_w, n)

            groups.append(
                ConsolidationGroup(
                    group_id=f"GRP-{counter:03d}",
                    cluster=int(cluster_id),
                    window_start=window_df["requested_delivery_date"].min().strftime("%Y-%m-%d"),
                    window_end=window_df["requested_delivery_date"].max().strftime("%Y-%m-%d"),
                    orders=orders,
                    total_weight_lbs=round(total_w, 1),
                    shipment_type=stype,
                )
            )
            counter += 1

    return groups
