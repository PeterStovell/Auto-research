"""
Pre-training script for the Switch Transformer backbone.

Uses M5 and UCI Electricity data (see src/pretrain/dataloader.py) to pre-train
the transformer backbone before fine-tuning on gas volume data.

Usage:
    python src/train_pretrain.py --output out/pretrain_run1
    python src/train_pretrain.py --output out/pretrain_run1 --max_epochs 30

After pre-training, load the backbone in fine-tuning:
    python src/train_switch.py --pretrain_backbone out/pretrain_run1/pretrain_backbone.pt --output out/finetune_run1
"""

import os
import sys
import argparse
import random

import pandas as pd
import torch
from omegaconf import OmegaConf

# Ensure src/ is on the path when running from repo root
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils import conf_write, get_accelerator, plot_curves, write_model_architecture
from train_switch import Model, TrainerConfig, ModelConfig, train_model
from pretrain.dataloader import build_pretrain_dataloaders


def main():
    print("python", sys.version)
    print("pytorch", torch.__version__)

    parser = argparse.ArgumentParser(description="Pre-train Switch Transformer backbone")
    parser.add_argument("--output", required=True, help="Path to output folder")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument("--data_dir", default="data/pretrain", help="Directory with pretrain parquets")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size")
    args = parser.parse_args()

    seed = 10
    trainer_cfg = OmegaConf.structured(TrainerConfig())
    model_cfg = OmegaConf.structured(ModelConfig())

    trainer_cfg.max_steps = 1000
    trainer_cfg.max_epochs = 5
    if args.max_epochs is not None:
        trainer_cfg.max_epochs = args.max_epochs
    if args.batch_size is not None:
        trainer_cfg.batch_size = args.batch_size

    random.seed(seed)
    torch.manual_seed(seed)

    print("Building pre-training dataloaders...")
    train_loader, val_loader, cat_card, n_num, n_target = build_pretrain_dataloaders(
        batch_size=trainer_cfg.batch_size,
        data_dir=args.data_dir,
    )

    device = get_accelerator(torch)
    print("device:", device)

    print("Creating model...")
    model = Model(cat_card, n_num, n_target, cfg=model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"cat_card={cat_card}, n_num={n_num}, n_target={n_target}")
    write_model_architecture(model)

    os.makedirs(args.output, exist_ok=True)

    print("Pre-training model...")
    metrics = train_model(
        model, train_loader, val_loader, trainer_cfg, device, args.output,
        loss_fn=lambda batch: model.compute_pretrain_loss(batch, device),
    )

    metrics_df = pd.DataFrame(metrics)
    metrics_indexed = metrics_df.set_index("epoch")

    run_summary = {
        "best_val_loss": float(metrics_df["val_loss"].min()),
        "best_epoch": int(metrics_indexed["val_loss"].idxmin()),
        "max_epochs": trainer_cfg.max_epochs,
        "curve": {k: v.tolist() for k, v in metrics_df.items()},
    }
    run_summary = OmegaConf.create(run_summary)
    conf_write(run_summary, os.path.join(args.output, "run_summary.yaml"))
    plot_curves(run_summary, os.path.join(args.output, "curves.png"))

    # Save backbone weights separately for fine-tuning transfer
    backbone_state = {
        k: v for k, v in model.state_dict().items()
        if k.startswith("layers.") or k.startswith("input_norm") or k.startswith("output_norm")
    }
    backbone_path = os.path.join(args.output, "pretrain_backbone.pt")
    torch.save(backbone_state, backbone_path)
    print(f"Backbone saved ({len(backbone_state)} keys): {backbone_path}")

    print(f"output={args.output}")


if __name__ == "__main__":
    main()
