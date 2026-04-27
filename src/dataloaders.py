import pandas as pd
import torch

from columns import Columns
from sampler import dataframe_to_sequence_list, SliceDataset
from encoder import get_encoder
from utils import conf_read


def build_dataloaders(batch_size):
    cfg = conf_read(__file__.replace('.py', '.yaml'))
    print('Loading ', cfg.input_path)
    columns = Columns(cfg.columns)  # add helper code
    df = pd.read_parquet(cfg.input_path, columns=columns.load_list()).dropna()
    if "query" in cfg:
        print("query:", cfg.query)
        before = len(df)
        df = df.query(cfg.query)
        after = len(df)
        print(f"dataset reduced by {1 - after / before : .2%}")
    print('Adding auxiliary columns')
    columns.add_auxiliary(df)
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

    print("Creating datasets")
    train_ds = SliceDataset(sequence_list, length=cfg.seq_len, end_date=cfg.train_end_date)
    print('training samples: ', len(train_ds))
    val_ds = SliceDataset(sequence_list, length=cfg.seq_len, start_date=cfg.train_end_date,
                          end_date=cfg.val_end_date)
    print('validation samples: ', len(val_ds))
    test_ds = SliceDataset(sequence_list, length=cfg.seq_len, start_date=cfg.val_end_date)
    print('test samples: ', len(test_ds))

    train_dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
    )

    cat_card = [df[c].nunique() for c in columns.categoricals()]
    n_num = len(columns.numericals())
    n_target = len(columns.targets())

    return train_dataloader, val_dataloader, test_dataloader, cat_card, n_num, n_target
