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

# Train/val split fractions (applied to the actual date range in the parquet).
# 80% of dates → train, next 10% → val. The final 10% is held out (not used).
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10


def _compute_split_dates(df: pd.DataFrame, date_col: str):
    """Compute train/val cutoff dates from the actual date range in df."""
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    total_days = (max_date - min_date).days
    train_end = min_date + pd.Timedelta(days=int(total_days * TRAIN_FRAC))
    val_end   = min_date + pd.Timedelta(days=int(total_days * (TRAIN_FRAC + VAL_FRAC)))
    print(f"  date range: {min_date.date()} → {max_date.date()} ({total_days} days)")
    print(f"  train: < {train_end.date()},  val: [{train_end.date()}, {val_end.date()})")
    return train_end, val_end


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


def _build_encoder(columns: Columns, df: pd.DataFrame, train_end_date):
    """
    Encoder for pretrain data:
    - Categoricals: OrdinalEncoder (dataset_id and group_id are already integers,
      but OrdinalEncoder ensures 0-indexed contiguous codes for nn.Embedding).
    - Numericals / targets: PassThrough (already log1p-scaled).
    """
    # 1. Create a categorical encoder wrapper around scikit-learn's OrdinalEncoder
    # We use handle_unknown="use_encoded_value" and unknown_value=np.nan 
    # so that any categories present in validation/test but NOT in training
    # will be encoded as NaN (and safely dropped or handled later).
    cat_enc = Wrapper(
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
        columns.encoder_list(),
    )
    
    # 2. Build the main composite Encoder
    # This orchestrates the transformations for different feature types:
    encoder = Encoder(
        # Categorical columns get ordinal-encoded to 0, 1, 2... for embedding layers
        categorical_transformer=cat_enc,
        # Numerical & target columns pass through unchanged (as they are already log1p scaled upstream)
        numerical_transformer=PassThrough(),
        target_transformer=PassThrough(),
    )
    
    # 3. Fit on train portion only
    # It is critical to only .fit() on training data to prevent data leakage.
    # The OrdinalEncoder will only "see" categories from the training set.
    train_mask = df[columns.date()] < train_end_date
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

    # Compute data-driven split dates
    TRAIN_END_DATE, VAL_END_DATE = _compute_split_dates(df, columns.date())

    # Determine cat cardinalities before encoding
    n_datasets = df["dataset_id"].nunique()   # 2
    n_groups   = df["group_id"].nunique()     # ≥10 (M5 has 7, electricity has 10, merged ~17)
    cat_card   = [n_datasets, n_groups]
    print(f"cat_card={cat_card}")

    print("Building encoder and transforming...")
    encoder = _build_encoder(columns, df, TRAIN_END_DATE)
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
            f"Split dates: train_end={TRAIN_END_DATE.date()}, val_end={VAL_END_DATE.date()}."
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
    )

    return train_loader, val_loader, cat_card, 2, 1
