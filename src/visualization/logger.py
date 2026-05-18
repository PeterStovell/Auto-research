"""
DiagnosticLogger: hooks into a SwitchTransformer model during training,
computes NAS regime signals each epoch, and appends them to a JSONL log file.

Signals collected
-----------------
NAS-01  effective_rank       SVD entropy of layer output activations
NAS-02  gradient_snr         |mean(grad)| / std(grad) per parameter
NAS-04  attn_entropy         Shannon entropy of attention weight distributions
NAS-07  dead_units           Fraction of activations with |x| < 1e-3
NAS-08  residual_acf1        Lag-1 autocorrelation of prediction residuals
NAS-12  sv_concentration     Fraction of variance in top-5 singular values
NAS-17  layer_redundancy     Linear CKA between consecutive layer outputs
NAS-19  calibration          Mean/std of predictions vs targets
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Signal computation primitives
# ---------------------------------------------------------------------------

def _effective_rank(x: torch.Tensor) -> float:
    """NAS-01: exp(H) of normalised singular value distribution."""
    x2d = x.detach().reshape(-1, x.shape[-1]).float()
    sv = torch.linalg.svdvals(x2d)
    sv = sv[sv > 1e-8]
    if sv.numel() < 2:
        return 1.0
    p = sv / sv.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()).item())


def _dead_unit_rate(x: torch.Tensor, threshold: float = 1e-3) -> float:
    """NAS-07: fraction of activations below threshold."""
    return float((x.detach().abs() < threshold).float().mean().item())


def _sv_concentration(x: torch.Tensor, k: int = 5) -> float:
    """NAS-12: fraction of total variance captured by the top-k singular values."""
    x2d = x.detach().reshape(-1, x.shape[-1]).float()
    sv = torch.linalg.svdvals(x2d)
    total = float((sv ** 2).sum().item()) + 1e-8
    return float((sv[:k] ** 2).sum().item() / total)


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """NAS-17: linear CKA similarity between two representation matrices."""
    X = X.detach().reshape(-1, X.shape[-1]).float()
    Y = Y.detach().reshape(-1, Y.shape[-1]).float()
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = float((X.T @ Y).pow(2).sum().item())
    hsic_xx = float((X.T @ X).pow(2).sum().item())
    hsic_yy = float((Y.T @ Y).pow(2).sum().item())
    return hsic_xy / (math.sqrt(hsic_xx * hsic_yy) + 1e-8)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class DiagnosticLogger:
    """
    Usage in training loop
    ----------------------
    logger = DiagnosticLogger(model, 'output/diagnostics.jsonl')

    # inside run_epoch_train, after loss.backward():
    logger.record_gradients()

    # after each eval phase, with a sample batch:
    logger.compute_and_log(epoch, pred, target)

    # at the end of training:
    logger.remove_hooks()
    """

    def __init__(self, model: nn.Module, log_path: str):
        self.model = model
        self.log_path = Path(log_path)
        self._layer_out: dict[str, torch.Tensor] = {}
        self._attn_weights: dict[str, torch.Tensor] = {}
        self._grad_snr_accum: dict[str, list] = {}
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'SwitchTransformerLayer':
                def _make(k):
                    def h(mod, inp, out):
                        self._layer_out[k] = out[0].detach()
                    return h
                self._hooks.append(module.register_forward_hook(_make(name)))

            if isinstance(module, nn.MultiheadAttention):
                def _make_attn(k):
                    def h(mod, inp, out):
                        if out[1] is not None:
                            self._attn_weights[k] = out[1].detach()
                    return h
                self._hooks.append(module.register_forward_hook(_make_attn(name)))

    def record_gradients(self):
        """Accumulate per-parameter gradient SNR. Call after loss.backward()."""
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                g = p.grad.detach().float()
                snr = float(g.abs().mean().item() / (g.std().item() + 1e-8))
                self._grad_snr_accum.setdefault(name, []).append(snr)

    def compute_and_log(self, epoch: int, pred: torch.Tensor, target: torch.Tensor):
        """
        Compute all signals from accumulated state + provided pred/target,
        then append one JSON record to the log file.

        pred, target: tensors from a representative eval batch, shape (B, n_target).
        """
        signals: dict = {}

        # NAS-01
        signals['NAS01_effective_rank'] = {
            k: _effective_rank(v) for k, v in self._layer_out.items()
        }

        # NAS-02: mean SNR over all backward steps this epoch
        signals['NAS02_gradient_snr'] = {
            k: float(sum(v) / len(v))
            for k, v in self._grad_snr_accum.items() if v
        }
        self._grad_snr_accum.clear()

        # NAS-04
        attn_ent = {}
        for k, w in self._attn_weights.items():
            w_flat = w.reshape(-1, w.shape[-1]).float()
            ent = float(-(w_flat * (w_flat + 1e-12).log()).sum(-1).mean().item())
            attn_ent[k] = ent
        signals['NAS04_attn_entropy'] = attn_ent

        # NAS-07
        signals['NAS07_dead_units'] = {
            k: _dead_unit_rate(v) for k, v in self._layer_out.items()
        }

        # NAS-08
        r = (pred.detach().float() - target.detach().float()).reshape(-1)
        if r.numel() > 2:
            r = r - r.mean()
            acf1 = float((r[:-1] * r[1:]).mean().item() / (r.var().item() + 1e-8))
        else:
            acf1 = 0.0
        signals['NAS08_residual_acf1'] = acf1

        # NAS-12
        signals['NAS12_sv_concentration'] = {
            k: _sv_concentration(v) for k, v in self._layer_out.items()
        }

        # NAS-17
        keys = list(self._layer_out.keys())
        redundancy = {}
        for i in range(len(keys) - 1):
            label = f'{keys[i]}__{keys[i + 1]}'
            redundancy[label] = _linear_cka(self._layer_out[keys[i]], self._layer_out[keys[i + 1]])
        signals['NAS17_layer_redundancy'] = redundancy

        # NAS-19
        p, t = pred.detach().float(), target.detach().float()
        signals['NAS19_calibration'] = {
            'pred_mean':   float(p.mean()),
            'pred_std':    float(p.std()),
            'target_mean': float(t.mean()),
            'target_std':  float(t.std()),
            'mae':         float((p - t).abs().mean()),
        }

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'a') as f:
            f.write(json.dumps({'epoch': epoch, 'signals': signals}) + '\n')

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
