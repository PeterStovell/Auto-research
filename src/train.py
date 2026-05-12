import os
import argparse
import sys
import random
import pandas as pd
import numpy as np
from dataclasses import dataclass
from omegaconf import OmegaConf
import torch
from torch import nn

from utils import conf_read, conf_write, get_accelerator, plot_curves
from dataloaders import build_dataloaders


@dataclass
class TrainerConfig:
    batch_size: int = 512
    lr: float = 0.001
    patience: int = 15
    max_epochs: int = 50
    gamma: float = 0.9
    weight_decay: float = 0.0
    max_norm: float = 0.0


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.4


class Model(torch.nn.Module):

    def __init__(self, cat_card: list[int], n_num: int, n_target: int, cfg):
        super().__init__()
        self.cfg = cfg
        emb_dims = [int(np.log(card) + 1) for card in cat_card]
        input_dim = sum(emb_dims) + n_num

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim, max_norm=1.0)
            for card, emb_dim in zip(cat_card, emb_dims)
        ])
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.linear = nn.Linear(cfg.hidden_dim, n_target)

    def forward(self, cat, num):
        x = [emb(cat[..., i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x + [num], dim=-1)
        y, _ = self.lstm(x)
        return nn.functional.softplus(self.linear(y))

    def compute_train_loss(self, batch, device):
        date, seq, cat, num, target = batch
        cat, num, target = cat.to(device), num.to(device), target.to(device)
        pred = self(cat, num)
        return torch.abs(pred - target).mean()

    def compute_eval_loss(self, batch, device):
        date, seq, cat, num, target = batch
        cat, num, target = cat.to(device), num.to(device), target.to(device)
        pred = self(cat, num)
        pred = pred[:, -1, :]
        target = target[:, -1, :]
        return torch.nn.functional.l1_loss(pred, target)


def run_epoch_train(model, dataloader, optimizer, trainer_cfg, device, ema_state=None, ema_decay=0.999):
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

        if ema_state is not None:
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
                else:
                    ema_state[k].copy_(v)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trainer_cfg.max_epochs)

    ema_decay = 0.999
    ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

    best_val = float("inf")
    patience_left = trainer_cfg.patience
    best_path = os.path.join(output_path, "best_model.pt")
    metrics = []

    for epoch in range(trainer_cfg.max_epochs):
        train_loss = run_epoch_train(
            model, train_loader, optimizer, trainer_cfg, device,
            ema_state=ema_state, ema_decay=ema_decay,
        )

        backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state)
        val_loss = run_epoch_eval(model, val_loader, device)
        model.load_state_dict(backup)

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
            torch.save(ema_state, best_path)
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
    parser.add_argument('--output', required=True, help='path to output folder')
    parser.add_argument('--max_epochs', type=int, default=None, help='override max_epochs in trainer config')
    args = parser.parse_args()

    seed = 10
    trainer_cfg = OmegaConf.structured(TrainerConfig())
    model_cfg = OmegaConf.structured(ModelConfig())

    if args.max_epochs is not None:
        trainer_cfg.max_epochs = args.max_epochs

    random.seed(seed)
    torch.manual_seed(seed)

    train_dataloader, val_dataloader, test_dataloader, cat_card, n_num, n_target = (
        build_dataloaders(trainer_cfg.batch_size)
    )

    device = get_accelerator(torch)
    print('device:', device)

    print("Creating model")
    model = Model(cat_card, n_num, n_target, cfg=model_cfg).to(device)

    os.makedirs(args.output, exist_ok=True)

    print("Training model")
    metrics = train_model(model, train_dataloader, val_dataloader, trainer_cfg, device, args.output)

    metrics = pd.DataFrame(metrics)
    metrics_indexed = metrics.set_index("epoch")

    run_summary = {
        "best_val_loss": float(metrics["val_loss"].min()),
        "best_epoch": int(metrics_indexed["val_loss"].idxmin()),
        "max_epochs": trainer_cfg.max_epochs,
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
    print(f"output={args.output}")


if __name__ == '__main__':
    main()
