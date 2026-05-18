"""
Pre-training dataloader for the Switch Transformer backbone.

Loads m5_daily.parquet and electricity_daily.parquet, concatenates them,
and returns DataLoaders compatible with the main training loop.

Batch format: (date, seq, cat, num, target)
  cat    (B, T, 2)  — [dataset_id, group_id]
  num    (B, T, 2)  — [value, value_lag1]
  target (B, T, 1)  — [target]

cat_card = [2, n_groups]  (n_groups inferred from data, ≥10)
n_num    = 2
n_target = 1
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Allow imports from the src/ directory (encoder, sampler, columns)
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from omegaconf import OmegaConf
from columns import Columns
from encoder import Encoder, Wrapper, PassThrough
from sampler import dataframe_to_sequence_list, SliceDataset
from sklearn.preprocessing import OrdinalEncoder


SEQ_LEN = 30

# Train/val split dates.
# Electricity covers 2011-2014; use 2014-01-01 as val start and 2014-03-31 as val end.
# M5 covers ~2011-06 to 2016-05; use the last ~10% of dates.
# We apply a single global split across the combined dataset:
#   train: date < 2014-01-01
#   val:   2014-01-01 <= date < 2014-04-01
TRAIN_END_DATE = "2014-01-01"
VAL_END_DATE   = "2014-04-01"


def _make_columns() -> Columns:
    """Build a Columns object matching the unified pretrain schema."""
    cfg = OmegaConf.create({
        "date": "date",
        "sequence": "sequence",
        "categoricals": ["dataset_id", "group_id"],
        "numericals": ["value", "value_lag1"],
        "targets": ["target"],
        "scaling": [],  # no group-scaling column needed
    })
    return Columns(cfg)


def _build_encoder(columns: Columns, df: pd.DataFrame):
    """
    Encoder for pretrain data:
    - Categoricals: OrdinalEncoder (dataset_id and group_id are already integers,
      but OrdinalEncoder ensures 0-indexed contiguous codes for nn.Embedding).
    - Numericals / targets: PassThrough (already log1p-scaled).
    """
    cat_enc = Wrapper(
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
        columns.encoder_list(),
    )
    encoder = Encoder(
        categorical_transformer=cat_enc,
        numerical_transformer=PassThrough(),
        target_transformer=PassThrough(),
    )
    # Fit on train portion only
    train_mask = df[columns.date()] < pd.to_datetime(TRAIN_END_DATE)
    encoder.fit(df[train_mask])
    return encoder


def build_pretrain_dataloaders(batch_size: int, data_dir: str = "data/pretrain"):
    """
    Returns (train_loader, val_loader, cat_card, n_num, n_target).

    cat_card = [2, n_groups]
    n_num    = 2
    n_target = 1
    """
    m5_path   = os.path.join(data_dir, "m5_daily.parquet")
    elec_path = os.path.join(data_dir, "electricity_daily.parquet")

    print(f"Loading {m5_path}...")
    df_m5 = pd.read_parquet(m5_path)
    print(f"  M5 shape: {df_m5.shape}")

    print(f"Loading {elec_path}...")
    df_elec = pd.read_parquet(elec_path)
    print(f"  Electricity shape: {df_elec.shape}")

    df = pd.concat([df_m5, df_elec], ignore_index=True)
    del df_m5, df_elec
    print(f"Combined shape: {df.shape}")

    columns = _make_columns()

    # Ensure correct dtypes for categoricals
    df["dataset_id"] = df["dataset_id"].astype(int)
    df["group_id"]   = df["group_id"].astype(int)
    df["date"]       = pd.to_datetime(df["date"])

    # Determine cat cardinalities before encoding
    n_datasets = df["dataset_id"].nunique()   # 2
    n_groups   = df["group_id"].nunique()     # ≥10 (M5 has 7, electricity has 10, merged ~17)
    cat_card   = [n_datasets, n_groups]
    print(f"cat_card={cat_card}")

    print("Building encoder and transforming...")
    encoder = _build_encoder(columns, df)
    df_t = encoder.transform(df)

    # Drop rows where encoding produced NaN (unknown categories)
    df_t = df_t.dropna(subset=columns.categoricals() + columns.numericals() + columns.targets())

    print("Building sequence list...")
    sequence_list = dataframe_to_sequence_list(df_t, columns)
    print(f"  {len(sequence_list):,} sequences")

    print("Creating SliceDatasets...")
    train_ds = SliceDataset(
        sequence_list, length=SEQ_LEN,
        start_date=None, end_date=TRAIN_END_DATE,
    )
    val_ds = SliceDataset(
        sequence_list, length=SEQ_LEN,
        start_date=TRAIN_END_DATE, end_date=VAL_END_DATE,
    )
    print(f"  train samples: {len(train_ds):,}")
    print(f"  val   samples: {len(val_ds):,}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"Empty pretrain split: train={len(train_ds)}, val={len(val_ds)}. "
            f"Check that parquet files cover dates before {VAL_END_DATE}."
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
    )

    return train_loader, val_loader, cat_card, 2, 1
