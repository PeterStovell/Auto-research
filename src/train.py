import os
import argparse
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn as nn

from sampler import dataframe_to_sequence_list, SliceDataset
from columns import Columns
from io_utils import conf_read
from torch_utils import get_accelerator


class Model(torch.nn.Module):

    def __init__(self, cat_card:list[int], n_num:int, n_target:int,
                 hidden_dim:int=64,
                 num_layers:int=2,
                 dropout:float=0.25,
                 ):
        super().__init__()
        emb_dims = [int(np.log(card) + 1) for card in cat_card]
        input_dim = sum(emb_dims) + n_num
        self.embeddings = nn.ModuleList([nn.Embedding(card, emb_dim)
                                         for card, emb_dim in zip(cat_card, emb_dims)])
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )
        self.linear = nn.Linear(in_features=hidden_dim, out_features=n_target)
        self.loss_function = torch.nn.L1Loss()


    def forward(self, cat, num):
        """
        cat: tensor of indices, shape = [batch_size, seq_len, n_cat] with n_cat = len(cat_card)
        num: tensor of floats, shape = [batch_size, seq_len, n_num]

        Note that the batch dimension is optional
        """

        # embedding of categories
        x = [emb(cat[..., i]) for i, emb in enumerate(self.embeddings)]
        # concatenate embeddings and numerical values along last axis
        x = torch.cat(x + [num], dim=-1)  # shape is [batch_size, seq_len, input_dim]
        y, _ = self.lstm(x)  # shape is [batch_size, seq_len, hidden_dim]
        y = self.linear(y)  # shape is [batch_size, seq_len, n_target]
        y = nn.functional.softplus(y)
        return y

    def compute_loss(self, batch, device):
        date, seq, cat, num, target = batch
        cat, num, target = cat.to(device), num.to(device), target.to(device)
        pred = self(cat, num)
        pred = pred[:, -1, :]
        target = target[:, -1, :]
        loss = self.loss_function(pred, target)
        return loss


def run_epoch_train(model, dataloader, max_steps_per_epoch=0, optimizer=None, device='cpu',
                    max_norm=0.,
                    ):
    total_loss = 0.
    n = 0
    model.train()
    for batch in dataloader:
        loss = model.compute_loss(batch, device=device)
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        total_loss += loss.item()
        n += 1
        if n == max_steps_per_epoch:
            break
    return total_loss / max(n, 1)


def run_epoch_eval(model, dataloader, device='cpu'):
    total_loss = 0.
    n = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            loss = model.compute_loss(batch, device=device)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    max_epochs,
    max_steps_per_epoch,
    optimizer_name,
    lr,
    weight_decay,
    gamma,
    max_norm,
    patience,
    device,
    output_path
):

    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    best_val = float("inf")
    patience_left = patience
    best_path = os.path.join(output_path, "best_model.pt")
    metrics = []
    for epoch in range(max_epochs):
        train_loss = run_epoch_train(model, train_loader,
                                     max_steps_per_epoch=max_steps_per_epoch,
                                     optimizer=optimizer, device=device, max_norm=max_norm)
        val_loss = run_epoch_eval(model, val_loader, device)
        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"lr={last_lr:.2e}"
        )
        metrics.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": last_lr,
        })
        if val_loss < best_val:
            print('*')
            best_val = val_loss
            patience_left = patience
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load(best_path))
    return metrics


def main(args: argparse.Namespace):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
    cfg = conf_read(config_path)
    print('Loading ', cfg.input_path)
    columns = Columns(cfg['columns'])  # add helper code
    df = pd.read_parquet(cfg.input_path, columns=columns.load_list()).dropna()
    print('Adding auxiliary columns')
    columns.add_auxiliary(df)
    print('shape:', df.shape)
    print('nas:')
    print(df.isna().sum())
    print('zeros:')
    print(df[columns.targets()].lt(1).mean())

    print('Encoding and scaling')
    from encoder import get_encoder
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
    seed = cfg.get('seed', 0)
    random.seed(seed)
    torch.manual_seed(seed)
    train_ds = SliceDataset(sequence_list, length=cfg.seq_len, end_date=cfg.train_end_date)
    print('training samples: ', len(train_ds))
    val_ds = SliceDataset(sequence_list, length=cfg.seq_len, start_date=cfg.train_end_date,
                          end_date=cfg.val_end_date)
    print('validation samples: ', len(val_ds))
    test_ds = SliceDataset(sequence_list, length=cfg.seq_len, start_date=cfg.val_end_date)
    print('test samples: ', len(test_ds))

    train_dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=0,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0,
    )

    print("Creating model")
    cat_card = [df[c].nunique() for c in columns.categoricals()]
    n_num = len(columns.numericals())
    n_target = len(columns.targets())
    device = cfg.get('accelerator', 'auto')
    if device == 'auto':
        device = get_accelerator()
    model = Model(
        cat_card, n_num, n_target,
    ).to(device)

    os.makedirs(args.output_path, exist_ok=True)

    print('device:', device)

    print("Training model")
    metrics = train_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        max_epochs=cfg.get('max_epochs', 50),
        max_steps_per_epoch=cfg.get('max_steps_per_epoch', -1),
        optimizer_name=cfg.get('optimizer_name', 'Adam'),
        lr=cfg.get('lr', 1e-3),
        weight_decay=cfg.get('weight_decay', 0.),
        gamma=cfg.get('gamma', 1.0),
        max_norm=cfg.get('max_norm', 0.),
        patience=cfg.get('patience', 10),
        device=device,
        output_path=args.output_path,
    )
    print(metrics)

    print("Testing model")
    test_loss = run_epoch_eval(model, test_dataloader, device=device)
    print(f"test_loss={test_loss:.5f}")


if __name__ == '__main__':
    print("python", sys.version)
    print("pytorch", torch.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default=f'./out/local', help='output path')
    parsed_args = parser.parse_args()
    main(parsed_args)
