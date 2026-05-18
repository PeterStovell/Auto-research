"""
Preprocess M5 Walmart retail sales data into the unified pre-training parquet schema.

Input (data/pretrain/raw/):
  - sales_train_validation.csv  (wide format: id, item_id, dept_id, ..., d_1..d_1913)
  - calendar.csv                (maps d_1..d_1913 to dates)

Output: data/pretrain/m5_daily.parquet

Unified schema:
  date        datetime64[ns]
  sequence    str              e.g. "m5__CA_1__FOODS_1__item_001"
  value       float64          log1p(sales)
  value_lag1  float64          log1p(sales at t-1)
  target      float64          log1p(sales at t+2)
  dataset_id  int8             0 for M5
  group_id    int16            dept_id encoded as 0..6
"""

import argparse
import os
import numpy as np
import pandas as pd

# 7 departments in M5
DEPT_ORDER = [
    "FOODS_1", "FOODS_2", "FOODS_3",
    "HOBBIES_1", "HOBBIES_2",
    "HOUSEHOLD_1", "HOUSEHOLD_2",
]
DEPT_TO_ID = {d: i for i, d in enumerate(DEPT_ORDER)}


def preprocess(raw_dir: str, out_dir: str):
    sales_path = os.path.join(raw_dir, "sales_train_validation.csv")
    calendar_path = os.path.join(raw_dir, "calendar.csv")

    print("Loading calendar...")
    calendar = pd.read_csv(calendar_path, usecols=["d", "date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    d_to_date = calendar.set_index("d")["date"].to_dict()

    print("Loading sales (wide format)...")
    # Read meta columns + day columns separately to avoid dtype issues
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales_wide = pd.read_csv(sales_path)

    # Build sequence ID: "m5__{store_id}__{dept_id}__{item_id}"
    sales_wide["sequence"] = (
        "m5__"
        + sales_wide["store_id"]
        + "__"
        + sales_wide["dept_id"]
        + "__"
        + sales_wide["item_id"]
    )

    dept_col = sales_wide["dept_id"].copy()

    # Day columns
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    print(f"Melting {len(sales_wide):,} rows x {len(day_cols)} days...")
    # Process store-by-store to keep memory manageable
    frames = []
    stores = sales_wide["store_id"].unique()
    for store in stores:
        mask = sales_wide["store_id"] == store
        sub = sales_wide.loc[mask, ["sequence", "dept_id"] + day_cols]
        melted = sub.melt(
            id_vars=["sequence", "dept_id"],
            value_vars=day_cols,
            var_name="d",
            value_name="raw_value",
        )
        melted["date"] = melted["d"].map(d_to_date)
        melted = melted.drop(columns=["d"])
        frames.append(melted)
        print(f"  store {store}: {len(melted):,} rows")

    print("Concatenating stores...")
    df = pd.concat(frames, ignore_index=True)
    del frames

    print("Applying log1p scaling...")
    df["raw_value"] = df["raw_value"].fillna(0.0).clip(lower=0.0)
    df["value"] = np.log1p(df["raw_value"])
    df = df.drop(columns=["raw_value"])

    print("Computing lag1 and target per sequence...")
    df = df.sort_values(["sequence", "date"])
    df["value_lag1"] = df.groupby("sequence", observed=True)["value"].shift(1)
    df["target"] = df.groupby("sequence", observed=True)["value"].shift(-2)

    print("Dropping NaN rows...")
    df = df.dropna(subset=["value_lag1", "target"])

    print("Assigning dataset_id and group_id...")
    df["dataset_id"] = np.int8(0)
    df["group_id"] = df["dept_id"].map(DEPT_TO_ID).astype(np.int16)

    # Final schema
    out = df[["date", "sequence", "value", "value_lag1", "target", "dataset_id", "group_id"]].copy()
    out = out.reset_index(drop=True)

    print(f"Final shape: {out.shape}")
    print(out.dtypes)
    print(out.head(3))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "m5_daily.parquet")
    out.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess M5 to unified parquet")
    parser.add_argument("--raw_dir", default="data/pretrain/raw", help="Directory with M5 CSVs")
    parser.add_argument("--out_dir", default="data/pretrain", help="Output directory")
    args = parser.parse_args()
    preprocess(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
