import os
import argparse
import sys
import random
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from dataclasses import dataclass
import torch
from torch import nn

from io_utils import conf_read, conf_write
from dataloaders import build_dataloaders
from torch_utils import get_accelerator


@dataclass
class DataLoaderConfig:
  seq_len: int = 12
  batch_size: int = 512


@dataclass
class TrainerConfig:
  max_epochs: int =  5
  patience: int = 15
  lr: float = 0.005
  weight_decay: float = 0. # active if positive
  gamma: float = 0.9
  max_norm: float = 0. # active if positive


@dataclass
class ModelConfig:
  hidden_dim: int = 128
  num_layers: int = 2
  dropout: float = 0.2


class Model(torch.nn.Module):

    def __init__(self, cat_card:list[int], n_num:int, n_target:int):
        super().__init__()
        emb_dims = [int(np.log(card) + 1) for card in cat_card]
        input_dim = sum(emb_dims) + n_num
        self.embeddings = nn.ModuleList([nn.Embedding(card, emb_dim)
                                         for card, emb_dim in zip(cat_card, emb_dims)])
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=ModelConfig.hidden_dim,
            num_layers=ModelConfig.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=ModelConfig.dropout,
        )
        self.linear = nn.Linear(in_features=ModelConfig.hidden_dim, out_features=n_target)
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


def run_epoch_train(model, dataloader, optimizer, device):
    total_loss = 0.
    n = 0
    model.train()
    for batch in dataloader:
        loss = model.compute_loss(batch, device=device)
        optimizer.zero_grad()
        loss.backward()
        if TrainerConfig.max_norm > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), TrainerConfig.max_norm)
        optimizer.step()
        total_loss += loss.item()
        n += 1
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
    device,
    output_path,
):

    optimizer = torch.optim.Adam(model.parameters(), lr=TrainerConfig.lr, weight_decay=TrainerConfig.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=TrainerConfig.gamma)

    best_val = float("inf")
    patience_left = TrainerConfig.patience
    best_path = os.path.join(output_path, "best_model.pt")
    metrics = []
    for epoch in range(TrainerConfig.max_epochs):
        train_loss = run_epoch_train(model, train_loader, optimizer, device)
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
            patience_left = TrainerConfig.patience
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load(best_path))
    return metrics


def main():
    print("python", sys.version)
    print("pytorch", torch.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default=f'./out/local', help='output path')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "static_config.yaml")
    cfg = conf_read(config_path)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataloader_config = OmegaConf.merge(cfg.dataloader, DataLoaderConfig)
    train_dataloader, val_dataloader, test_dataloader, cat_card, n_num, n_target = build_dataloaders(dataloader_config)

    device = get_accelerator()
    print('device:', device)

    print("Creating model")
    model = Model(cat_card, n_num, n_target).to(device)

    os.makedirs(args.output_path, exist_ok=True)

    print("Training model")
    metrics = train_model(model, train_dataloader, val_dataloader, device, args.output_path)
    metrics = pd.DataFrame(metrics)
    print(metrics.set_index("epoch"))

    run_summary = {
        "curve": {k: v.tolist() for k, v in metrics.items()},
    }
    run_summary = OmegaConf.create(run_summary)
    conf_write(run_summary, os.path.join(args.output_path, 'run_summary.yaml'))

    print("Testing model")
    test_loss = run_epoch_eval(model, test_dataloader, device=device)
    print(f"test_loss={test_loss:.5f}")


if __name__ == '__main__':
    main()
