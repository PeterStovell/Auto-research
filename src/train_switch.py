import os
import math
import time
import argparse
import sys
import random
import pandas as pd
import numpy as np
from dataclasses import dataclass
from omegaconf import OmegaConf
import torch
from torch import nn
import torch.nn.functional as F

from utils import conf_read, conf_write, get_accelerator, plot_curves, write_model_architecture
from dataloaders import build_dataloaders
from visualization import DiagnosticLogger, plot_all


@dataclass
class TrainerConfig:
    batch_size: int = 48
    lr: float = 0.001
    patience: int = 15
    min_delta: float = 0.0001
    max_epochs: int = 50
    max_seconds: float = -1.0
    max_steps: int = -1
    weight_decay: float = 0.0
    max_norm: float = 1.0


@dataclass
class ModelConfig:
    hidden_dim: int = 64      # d_model / transformer width
    num_layers: int = 4         # number of transformer blocks (must be even: Dense/Switch alternating)
    lstm_layers: int = 2        # LSTM layers before the transformer stack
    num_heads: int = 4          # multi-head attention heads
    num_experts: int = 8        # regime experts per Switch FFN layer
    ffn_dim: int = 128          # hidden dim inside each expert FFN
    capacity_factor: float = 1.25  # overflow buffer for expert dispatch
    dropout: float = 0.1
    aux_loss_weight: float = 0.1   # weight for load-balancing auxiliary loss
    multistep_steps: int = 3       # steps ahead for multi-step forecast pretrain task
    impute_mask_ratio: float = 0.15  # fraction of timesteps masked for imputation pretrain task
    use_raw_skip: bool = False     # if True, feed projected raw features as skip into every Switch layer


# ---------------------------------------------------------------------------
# Switch Transformer building blocks
# ---------------------------------------------------------------------------

class SwitchExpert(nn.Module):
    """A single regime expert: a two-layer FFN with GELU activation."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class SwitchFFN(nn.Module):
    """
    Switch Feed-Forward layer.

    Each token is routed to exactly one expert (k=1 top-1 routing).
    Tokens that exceed an expert's capacity are left unmodified (zero
    contribution to the residual stream, handled by the residual connection).

    Also computes the auxiliary load-balancing loss:
        L_aux = E * sum_i(f_i * P_i)
    where f_i = fraction of tokens dispatched to expert i,
          P_i = mean router probability for expert i.
    This encourages uniform usage across experts.
    """

    def __init__(self, d_model: int, ffn_dim: int, num_experts: int,
                 capacity_factor: float, dropout: float):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwitchExpert(d_model, ffn_dim, dropout) for _ in range(num_experts)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)                                        # (N, D)

        # Router probabilities
        router_probs = torch.softmax(self.router(x_flat), dim=-1)    # (N, E)

        # Top-1 dispatch
        expert_idx = router_probs.argmax(dim=-1)                     # (N,)
        expert_gate = router_probs[torch.arange(N, device=x.device), expert_idx]  # (N,)

        # Auxiliary load-balancing loss
        tokens_per_expert = torch.bincount(expert_idx, minlength=self.num_experts).float()
        f = tokens_per_expert / N                                     # fraction dispatched
        P = router_probs.mean(dim=0)                                  # mean router prob
        aux_loss = self.num_experts * (f * P).sum()

        # Per-expert capacity limit
        capacity = max(1, int(self.capacity_factor * N / self.num_experts))

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_indices = (expert_idx == i).nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue
            token_indices = token_indices[:capacity]                  # enforce capacity
            gates = expert_gate[token_indices].unsqueeze(-1)          # (k, 1)
            output[token_indices] = gates * expert(x_flat[token_indices])

        return self.dropout(output).view(B, T, D), aux_loss


class SwitchTransformerLayer(nn.Module):
    """
    One layer: causal self-attention + Switch FFN, both with pre-LayerNorm
    and residual connections.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int,
                 num_experts: int, capacity_factor: float, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.switch_ffn = SwitchFFN(d_model, ffn_dim, num_experts, capacity_factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor, skip: torch.Tensor = None):
        # Causal self-attention (pre-norm)
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + self.dropout(attn_out)

        # Inject raw-feature skip before Switch FFN
        if skip is not None:
            x = x + skip

        # Switch FFN (pre-norm)
        normed = self.norm2(x)
        ffn_out, aux_loss = self.switch_ffn(normed)
        x = x + ffn_out

        return x, aux_loss


class DenseTransformerLayer(nn.Module):
    """One block: causal self-attention + plain dense FFN, pre-LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor):
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x, x.new_zeros(())


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    Switch Transformer for regime-aware time series forecasting.

    Replaces the LSTM backbone with:
    - A linear input projection into d_model space
    - N layers of causal self-attention + Switch FFN
    - Each Switch FFN has `num_experts` independent expert networks,
      one per market regime (e.g. conflict, depression, expansion, …)
    - Top-1 routing: each time-step token is dispatched to one expert
    - Auxiliary load-balancing loss prevents expert collapse
    """

    def __init__(self, cat_card: list[int], n_num: int, n_target: int, cfg):
        super().__init__()
        self.cfg = cfg

        # Categorical embeddings (same scheme as baseline)
        emb_dims = [int(np.log(card) + 1) for card in cat_card]
        input_dim = sum(emb_dims) + n_num
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim, max_norm=1.0)
            for card, emb_dim in zip(cat_card, emb_dims)
        ])

        # LSTM input layer: captures immediate sequential dependencies
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.input_norm = nn.LayerNorm(cfg.hidden_dim)


        # Alternating Dense / Switch transformer blocks
        self.layers = nn.ModuleList()
        for i in range(cfg.num_layers):
            if i % 2 == 0:
                self.layers.append(DenseTransformerLayer(
                    d_model=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    ffn_dim=cfg.ffn_dim,
                    dropout=cfg.dropout,
                ))
            else:
                self.layers.append(SwitchTransformerLayer(
                    d_model=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    ffn_dim=cfg.ffn_dim,
                    num_experts=cfg.num_experts,
                    capacity_factor=cfg.capacity_factor,
                    dropout=cfg.dropout,
                ))

        self.output_norm = nn.LayerNorm(cfg.hidden_dim)
        self.linear = nn.Linear(cfg.hidden_dim, n_target)
        self.multistep_head = nn.Linear(cfg.hidden_dim, cfg.multistep_steps)

    def _causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Additive float mask: future positions get -inf, past/current get 0."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1,
        )
        return mask

    def _encode(self, cat: torch.Tensor, num: torch.Tensor):
        """Returns hidden states (B, T, d_model) and aux_loss."""
        x = [emb(cat[..., i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x + [num], dim=-1)               # (B, T, input_dim)

        x, _ = self.lstm(x)                            # (B, T, hidden_dim)
        x = self.input_norm(x)                         # (B, T, d_model)

        # LSTM output is held as skip, injected into every Switch layer if enabled
        x_lstm = x if self.cfg.use_raw_skip else None

        causal_mask = self._causal_mask(x.shape[1], x.device)

        total_aux = x.new_zeros(())
        for layer in self.layers:
            skip = x_lstm if isinstance(layer, SwitchTransformerLayer) else None
            x, aux = layer(x, causal_mask, skip=skip)
            total_aux = total_aux + aux

        return self.output_norm(x), total_aux

    def _forward_internal(self, cat: torch.Tensor, num: torch.Tensor):
        h, aux = self._encode(cat, num)
        pred = F.softplus(self.linear(h))               # (B, T, n_target)
        return pred, aux

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        pred, _ = self._forward_internal(cat, num)
        return pred

    def compute_train_loss(self, batch, device):
        date, seq, cat, num, target = batch
        cat, num, target = cat.to(device), num.to(device), target.to(device)
        pred, aux_loss = self._forward_internal(cat, num)
        main_loss = torch.abs(pred - target).mean()
        return main_loss + self.cfg.aux_loss_weight * aux_loss

    def compute_pretrain_loss(self, batch, device):
        """Joint imputation + multi-step forecast loss for pretrain stage."""
        date, seq, cat, num, target = batch
        cat, num, target = cat.to(device), num.to(device), target.to(device)
        B, T, _ = num.shape

        # Task 1: Imputation — mask random timesteps, predict target at all positions
        mask = torch.rand(B, T, device=device) < self.cfg.impute_mask_ratio
        num_masked = num.clone()
        num_masked[mask.unsqueeze(-1).expand_as(num)] = 0.0
        h_imp, aux1 = self._encode(cat, num_masked)
        impute_loss = torch.abs(F.softplus(self.linear(h_imp)) - target).mean()

        # Task 2: Multi-step forecast — at position t predict target[t+1..t+k]
        k = self.cfg.multistep_steps
        h_ms, aux2 = self._encode(cat, num)
        ms_pred = F.softplus(self.multistep_head(h_ms))  # (B, T, k)
        if T > k:
            ms_target = torch.stack(
                [target[:, s:T - k + s, 0] for s in range(1, k + 1)], dim=-1
            )  # (B, T-k, k)
            ms_loss = torch.abs(ms_pred[:, :T - k, :] - ms_target).mean()
        else:
            ms_loss = ms_pred.new_zeros(())

        aux_loss = (aux1 + aux2) / 2
        return impute_loss + ms_loss + self.cfg.aux_loss_weight * aux_loss

    def compute_eval_loss(self, batch, device):
        date, seq, cat, num, target = batch
        cat, num, target = cat.to(device), num.to(device), target.to(device)
        pred = self(cat, num)
        pred = pred[:, -1, :]
        target = target[:, -1, :]
        return torch.nn.functional.l1_loss(pred, target)


# ---------------------------------------------------------------------------
# Training loop (identical to baseline)
# ---------------------------------------------------------------------------

def run_epoch_train(model, dataloader, optimizer, trainer_cfg, device,
                    ema_state=None, ema_decay=0.999, diag_logger=None, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda batch: model.compute_train_loss(batch, device=device)
    total_loss = 0.
    n = 0
    model.train()

    for batch in dataloader:
        loss = loss_fn(batch)

        optimizer.zero_grad()
        loss.backward()

        if trainer_cfg.max_norm > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_cfg.max_norm)

        if diag_logger is not None:
            diag_logger.record_gradients()

        optimizer.step()

        if ema_state is not None:
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
                else:
                    ema_state[k].copy_(v)

        total_loss += loss.item()
        n += 1
        if trainer_cfg.max_steps > 0 and n >= trainer_cfg.max_steps:
            break
        if n % 500 == 0:
            print(f"  step {n} loss={total_loss/n:.5f}", flush=True)

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


def train_model(model, train_loader, val_loader, trainer_cfg, device, output_path, loss_fn=None):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainer_cfg.lr,
        weight_decay=trainer_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    ema_decay = 0.999
    ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

    best_val = float("inf")
    patience_left = trainer_cfg.patience
    t_start = time.time()
    best_path = os.path.join(output_path, "best_model.pt")
    metrics = []

    # diag_logger = DiagnosticLogger(model, os.path.join(output_path, 'diagnostics.jsonl'))
    # diag_batch = next(iter(val_loader))  # fixed batch for consistent diagnostic snapshots

    for epoch in range(trainer_cfg.max_epochs):
        train_loss = run_epoch_train(
            model, train_loader, optimizer, trainer_cfg, device,
            ema_state=ema_state, ema_decay=ema_decay, loss_fn=loss_fn,
        )

        backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema_state)
        val_loss = run_epoch_eval(model, val_loader, device)

        # diagnostic snapshot using EMA model on the fixed val batch
        # date, seq, cat, num, target = diag_batch
        # cat, num, target = cat.to(device), num.to(device), target.to(device)
        # with torch.no_grad():
        #     pred = model(cat, num)
        # diag_logger.compute_and_log(epoch, pred[:, -1, :], target[:, -1, :])

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

        if trainer_cfg.max_seconds > 0 and time.time() - t_start > trainer_cfg.max_seconds:
            print(f"Time limit reached ({trainer_cfg.max_seconds}s), stopping pretrain")
            break

        if val_loss < best_val - trainer_cfg.min_delta:
            print('*')
            best_val = val_loss
            patience_left = trainer_cfg.patience
            torch.save(ema_state, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered")
                break

    # diag_logger.remove_hooks()
    # plot_all(
    #     os.path.join(output_path, 'diagnostics.jsonl'),
    #     os.path.join(output_path, 'diagnostics'),
    # )

    model.load_state_dict(torch.load(best_path))
    return metrics


def load_pretrained_backbone(model: nn.Module, backbone_path: str):
    """
    Load backbone weights from a pre-training run into the model.

    Only layers whose keys match (layers.*, input_norm*, output_norm*) are loaded.
    Mismatched heads (embeddings, input_proj, linear) are left randomly initialized.
    Uses strict=False so missing/unexpected keys are tolerated.
    """
    state = torch.load(backbone_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded backbone: {len(state)} keys, {len(missing)} missing, {len(unexpected)} unexpected")


def main():
    print("python", sys.version)
    print("pytorch", torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='path to output folder')
    parser.add_argument('--max_epochs', type=int, default=None, help='override max_epochs in trainer config')
    parser.add_argument('--pretrain_backbone', default=None, help='path to pretrain_backbone.pt for weight transfer')
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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    write_model_architecture(model)

    if args.pretrain_backbone is not None:
        print(f"Loading pretrained backbone from {args.pretrain_backbone}")
        load_pretrained_backbone(model, args.pretrain_backbone)

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

    print(f"output={args.output}")


if __name__ == '__main__':
    main()
