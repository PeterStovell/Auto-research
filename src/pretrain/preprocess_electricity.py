"""
Preprocess UCI ElectricityLoadDiagrams 2011-2014 into the unified pre-training parquet schema.

Input (data/pretrain/raw/):
  - LD2011_2014.txt  (semicolon-separated, decimal=',', 370 client columns, 15-min rows)

Output: data/pretrain/electricity_daily.parquet

Unified schema:
  date        datetime64[ns]
  sequence    str              e.g. "elec__MT_001"
  value       float64          log1p(daily kWh)
  value_lag1  float64          log1p(daily kWh at t-1)
  target      float64          log1p(daily kWh at t+2)
  dataset_id  int8             1 for electricity
  group_id    int16            0..9 (decile of mean daily load)
"""

import argparse
import os
import numpy as np
import pandas as pd


def preprocess(raw_dir: str, out_dir: str):
    txt_path = os.path.join(raw_dir, "LD2011_2014.txt")

    print("Loading electricity data (15-min resolution)...")
    df_raw = pd.read_csv(
        txt_path,
        sep=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )
    # Index is datetime, columns are MT_001..MT_370
    df_raw.index.name = "datetime"
    print(f"Loaded shape: {df_raw.shape}")

    print("Resampling to daily by summing (96 readings per day)...")
    df_daily = df_raw.resample("D").sum()
    print(f"Daily shape: {df_daily.shape}")

    print("Melting to long format...")
    df_daily = df_daily.reset_index()
    df_daily = df_daily.rename(columns={"datetime": "date"})
    df_long = df_daily.melt(id_vars="date", var_name="client", value_name="raw_value")

    # Sequence ID: "elec__MT_001"
    df_long["sequence"] = "elec__" + df_long["client"]
    df_long = df_long.drop(columns=["client"])

    print("Applying log1p scaling...")
    df_long["raw_value"] = df_long["raw_value"].fillna(0.0).clip(lower=0.0)
    df_long["value"] = np.log1p(df_long["raw_value"])

    print("Sorting and computing lag1/target per sequence...")
    df_long = df_long.sort_values(["sequence", "date"])
    df_long["value_lag1"] = df_long.groupby("sequence", observed=True)["value"].shift(1)
    df_long["target"] = df_long.groupby("sequence", observed=True)["value"].shift(-2)

    print("Dropping NaN rows...")
    df_long = df_long.dropna(subset=["value_lag1", "target"])

    print("Assigning dataset_id=1 and group_id (decile of mean daily load)...")
    df_long["dataset_id"] = np.int8(1)

    # Compute mean daily load per client for clustering
    mean_load = df_long.groupby("sequence", observed=True)["raw_value"].mean()
    decile_labels = pd.qcut(mean_load, q=10, labels=False, duplicates="drop")
    decile_map = decile_labels.to_dict()
    df_long["group_id"] = df_long["sequence"].map(decile_map).astype(np.int16)

    # Final schema
    out = df_long[["date", "sequence", "value", "value_lag1", "target", "dataset_id", "group_id"]].copy()
    out = out.reset_index(drop=True)

    print(f"Final shape: {out.shape}")
    print(out.dtypes)
    print(out.head(3))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "electricity_daily.parquet")
    out.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess UCI Electricity to unified parquet")
    parser.add_argument("--raw_dir", default="data/pretrain/raw", help="Directory with LD2011_2014.txt")
    parser.add_argument("--out_dir", default="data/pretrain", help="Output directory")
    args = parser.parse_args()
    preprocess(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
