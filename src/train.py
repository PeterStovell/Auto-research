import os
import argparse
import sys
import random
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from io_utils import conf_read, conf_write
from dataloaders import build_dataloaders
from torch_utils import get_accelerator
from plot_utils import plot_curves


class Model(torch.nn.Module):

    def __init__(self, cat_card: list[int], n_num: int, n_target: int, cfg):
        super().__init__()
        self.cfg = cfg
        emb_dims = [int(np.log(card) + 1) for card in cat_card]
        input_dim = sum(emb_dims) + n_num

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim)
            for card, emb_dim in zip(cat_card, emb_dims)
        ])
        self.input_proj = nn.Linear(input_dim, cfg.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.nhead,
            dim_feedforward=cfg.hidden_dim * 2,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.linear = nn.Linear(cfg.hidden_dim, n_target)

    def forward(self, cat, num):
        x = [emb(cat[..., i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x + [num], dim=-1)

        y = self.input_proj(x)
        mask = nn.Transformer.generate_square_subsequent_mask(y.size(1), device=y.device)
        y = self.transformer(y, mask=mask, is_causal=True)
        y = self.linear(y)
        return nn.functional.softplus(y)

    def compute_train_loss(self, batch, device):
        date, seq, cat, num, target = batch
        cat, num, target = cat.to(device), num.to(device), target.to(device)

        pred = self(cat, num)
        loss = torch.abs(pred - target)

        if self.cfg.use_weighted_loss:
            T = loss.size(1)
            weights = torch.linspace(0.2, 1.0, steps=T, device=loss.device).view(1, T, 1)
            loss = loss * weights

        return loss.mean()

    def compute_eval_loss(self, batch, device):
        date, seq, cat, num, target = batch
        cat, num, target = cat.to(device), num.to(device), target.to(device)
        pred = self(cat, num)
        pred = pred[:, -1, :]
        target = target[:, -1, :]
        return torch.nn.functional.l1_loss(pred, target)


def run_epoch_train(model, dataloader, optimizer, trainer_cfg, device):
    total_loss = 0.
    n = 0
    model.train()

    for batch in dataloader:
        loss = model.compute_train_loss(batch, device=device)

        optimizer.zero_grad()
        loss.backward()

        if trainer_cfg.max_norm > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_cfg.max_norm)

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
            loss = model.compute_eval_loss(batch, device=device)
            total_loss += loss.item()
            n += 1

    return total_loss / max(n, 1)


def train_model(model, train_loader, val_loader, trainer_cfg, device, output_path):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainer_cfg.lr,
        weight_decay=trainer_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trainer_cfg.gamma)

    best_val = float("inf")
    patience_left = trainer_cfg.patience
    best_path = os.path.join(output_path, "best_model.pt")
    metrics = []

    for epoch in range(trainer_cfg.max_epochs):
        train_loss = run_epoch_train(model, train_loader, optimizer, trainer_cfg, device)
        val_loss = run_epoch_eval(model, val_loader, device)

        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"lr={last_lr:.2e}"
        )

        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": last_lr})

        if val_loss < best_val:
            print('*')
            best_val = val_loss
            patience_left = trainer_cfg.patience
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
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--output', required=True, help='path to output folder')
    args = parser.parse_args()

    cfg = conf_read(args.config)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_dataloader, val_dataloader, test_dataloader, cat_card, n_num, n_target = \
        build_dataloaders(cfg.dataloader)

    device = get_accelerator()
    print('device:', device)

    print("Creating model")
    model = Model(cat_card, n_num, n_target, cfg=cfg.model).to(device)

    os.makedirs(args.output, exist_ok=True)

    print("Training model")
    metrics = train_model(model, train_dataloader, val_dataloader, cfg.trainer, device, args.output)

    metrics = pd.DataFrame(metrics)
    metrics_indexed = metrics.set_index("epoch")

    run_summary = {
        "best_val_loss": float(metrics["val_loss"].min()),
        "best_epoch": int(metrics_indexed["val_loss"].idxmin()),
        "max_epochs": cfg.max_epochs,
        "curve": {k: v.tolist() for k, v in metrics.items()},
    }
    run_summary = OmegaConf.create(run_summary)
    conf_write(run_summary, os.path.join(args.output, 'run_summary.yaml'))

    plot_curves(run_summary, os.path.join(args.output, 'curves.png'))

    # test_loss = run_epoch_eval(model, test_dataloader, device=device)
    #
    # print("\nSummary:")
    # print(f"  best_val_loss={run_summary.min_val_loss:.5f}  (epoch {run_summary.min_epoch} / {cfg.trainer.max_epochs})")
    # print(f"  test_loss   ={test_loss:.5f}")
    # print(f"  output      ={args.output}")


if __name__ == '__main__':
    main()
