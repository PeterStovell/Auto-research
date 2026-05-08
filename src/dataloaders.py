import os

import pandas as pd
import torch

from columns import Columns
from sampler import dataframe_to_sequence_list, SliceDataset
from encoder import get_encoder
from features import add_features
from utils import conf_read


def _cache_path(cfg):
    path = cfg.get("sequence_cache_path", None)
    if path is None:
        return None
    return os.path.expanduser(str(path))


def _load_sequence_cache(path):
    if path is None or not os.path.exists(path):
        return None

    print("Loading sequence cache", path)
    return torch.load(path, weights_only=False)


def _save_sequence_cache(path, payload):
    if path is None:
        return

    print("Saving sequence cache", path)
    cache_dir = os.path.dirname(path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    torch.save(payload, path)


def _build_sequence_cache(cfg, columns):
    print('Loading ', cfg.input_path)
    df = pd.read_parquet(cfg.input_path, columns=columns.load_list()).dropna()
    if "query" in cfg:
        print("query:", cfg.query)
        before = len(df)
        df = df.query(cfg.query)
        after = len(df)
        print(f"dataset reduced by {1 - after / before : .2%}")
    print('Adding auxiliary columns')
    columns.add_auxiliary(df)
    print('Adding engineered features')
    new_features = add_features(df, columns)
    columns['numericals'] = list(columns.numericals()) + new_features
    print(f'numericals ({len(columns.numericals())}):', list(columns.numericals()))
    print('shape:', df.shape)
    print('nas:')
    print(df.isna().sum())
    print('zeros:')
    print(df[columns.targets()].lt(1).mean())

    print('Encoding and scaling')
    encoder = get_encoder(columns)
    df_train_val = df.loc[df[columns.date()] < pd.to_datetime(cfg.val_end_date)]
    encoder.fit(df_train_val)
    train_val_sequences = df_train_val[columns.sequence()].unique()
    # keeping only the sequences known at train or val time
    df = df.loc[df[columns.sequence()].isin(train_val_sequences)]
    print('shape after removing unknown sequences:', df.shape)
    df_t = encoder.transform(df)
    print('nas:')
    print(df.isna().sum())
    print('uniques:')
    print(df.nunique())

    print('Transforming dataframe to sequence_list')
    sequence_list = dataframe_to_sequence_list(df_t, columns)

    return {
        "sequence_list": sequence_list,
        "cat_card": [df[c].nunique() for c in columns.categoricals()],
        "n_num": len(columns.numericals()),
        "n_target": len(columns.targets()),
    }


def _get_sequence_payload(cfg, columns):
    path = _cache_path(cfg)
    payload = _load_sequence_cache(path)
    if payload is not None:
        return payload

    payload = _build_sequence_cache(cfg, columns)
    _save_sequence_cache(path, payload)
    return payload


def build_dataloaders(batch_size):
    cfg = conf_read(__file__.replace('.py', '.yaml'))
    columns = Columns(cfg.columns)  # add helper code
    payload = _get_sequence_payload(cfg, columns)
    sequence_list = payload["sequence_list"]

    print("Creating datasets")
    train_ds = SliceDataset(
        sequence_list, length=cfg.seq_len,
        start_date=cfg.train_start_date, end_date=cfg.train_end_date,
    )
    print('training samples: ', len(train_ds))
    val_ds = SliceDataset(sequence_list, length=cfg.seq_len, start_date=cfg.train_end_date,
                          end_date=cfg.val_end_date)
    print('validation samples: ', len(val_ds))
    test_ds = SliceDataset(sequence_list, length=cfg.seq_len, start_date=cfg.val_end_date)
    print('test samples: ', len(test_ds))

    empty_splits = [name for name, ds in (("train", train_ds), ("validation", val_ds)) if len(ds) == 0]
    if empty_splits:
        raise ValueError(
            f"Empty dataset split(s): {', '.join(empty_splits)}. "
            f"Check train_start_date={cfg.train_start_date}, train_end_date={cfg.train_end_date}, val_end_date={cfg.val_end_date}, "
            f"and seq_len={cfg.seq_len}."
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
    )

    return (
        train_dataloader, val_dataloader, test_dataloader,
        payload["cat_card"], payload["n_num"], payload["n_target"],
    )
