"""
Preprocess orders data for freight route optimization.

Usage:
    python preprocess.py [orders.csv] [kmeans|dbscan]

Outputs a preprocessed CSV file with geocoded, clustered, and consolidated data.
"""

import sys
import pandas as pd
import os

from src.clustering import cluster_orders
from src.consolidation import consolidate_orders
from src.geolocation import geocode_orders


def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/sample_orders.csv"
    cluster_method = sys.argv[2] if len(sys.argv) > 2 else "kmeans"
    out_file = os.path.splitext(data_file)[0] + f"_{cluster_method}_preproc.csv"

    print(f"Loading orders from {data_file} ...")
    df = pd.read_csv(data_file)

    print("Geocoding cities ...")
    df_geo = geocode_orders(df)

    print(f"Clustering orders ({cluster_method}) ...")
    df_geo = cluster_orders(df_geo, method=cluster_method)

    print("Consolidating orders ...")
    groups = consolidate_orders(df_geo)

    # Flatten group info into DataFrame for CSV output
    group_rows = []
    for g in groups:
        for order in g.orders:
            row = order.copy()
            row.update(
                {
                    "group_id": g.group_id,
                    "cluster": g.cluster,
                    "window_start": g.window_start,
                    "window_end": g.window_end,
                    "total_weight_lbs": g.total_weight_lbs,
                    "shipment_type": g.shipment_type,
                }
            )
            group_rows.append(row)
    df_out = pd.DataFrame(group_rows)
    df_out.to_csv(out_file, index=False)
    print(f"Preprocessing complete. Saved to {out_file}.")


if __name__ == "__main__":
    main()
