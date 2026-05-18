"""
Read a diagnostics JSONL log (written by DiagnosticLogger) and save
one PNG per NAS signal to an output directory.

Entry point
-----------
    from visualization import plot_all
    plot_all('output/diagnostics.jsonl', 'output/diagnostics/')

Or run directly:
    python -m visualization.plots <log_path> <output_dir>
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_log(log_path: str) -> list[dict]:
    """Load a JSONL diagnostics log, sorted by epoch."""
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return sorted(records, key=lambda r: r['epoch'])


def _epochs(records: list[dict]) -> list[int]:
    return [r['epoch'] for r in records]


def _save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-signal plot functions  (all accept records + output_path)
# ---------------------------------------------------------------------------

def plot_nas01_rank_collapse(records: list[dict], output_path: str):
    """NAS-01: Effective rank of each layer's output activations over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = _epochs(records)
    for k in records[0]['signals']['NAS01_effective_rank']:
        vals = [r['signals']['NAS01_effective_rank'].get(k) for r in records]
        ax.plot(epochs, vals, label=k)
    ax.set(xlabel='Epoch', ylabel='Effective Rank',
           title='NAS-01: Layer Effective Rank (higher = more expressive)')
    ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_nas02_gradient_snr(records: list[dict], output_path: str):
    """NAS-02: Gradient SNR (mean ± 1σ across all parameters) over epochs."""
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = _epochs(records)
    all_snr = np.array([
        list(r['signals']['NAS02_gradient_snr'].values()) for r in records
    ], dtype=float)
    mean = np.nanmean(all_snr, axis=1)
    std = np.nanstd(all_snr, axis=1)
    ax.plot(epochs, mean, label='mean SNR')
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, label='±1σ')
    ax.set(xlabel='Epoch', ylabel='Gradient SNR',
           title='NAS-02: Gradient SNR across parameters')
    ax.legend()
    _save(fig, output_path)


def plot_nas04_attn_entropy(records: list[dict], output_path: str):
    """NAS-04: Attention entropy per layer over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = _epochs(records)
    for k in records[0]['signals']['NAS04_attn_entropy']:
        vals = [r['signals']['NAS04_attn_entropy'].get(k) for r in records]
        ax.plot(epochs, vals, label=k)
    ax.set(xlabel='Epoch', ylabel='Entropy (nats)',
           title='NAS-04: Attention Entropy (collapse → 0, uniform → log T)')
    ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_nas07_dead_units(records: list[dict], output_path: str):
    """NAS-07: Fraction of near-zero activations per layer over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = _epochs(records)
    for k in records[0]['signals']['NAS07_dead_units']:
        vals = [r['signals']['NAS07_dead_units'].get(k) for r in records]
        ax.plot(epochs, vals, label=k)
    ax.set(xlabel='Epoch', ylabel='Dead Unit Rate',
           title='NAS-07: Dead Unit Rate (|act| < 1e-3)')
    ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_nas08_acf(records: list[dict], output_path: str):
    """NAS-08: Lag-1 autocorrelation of prediction residuals over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [r['signals'].get('NAS08_residual_acf1', float('nan')) for r in records]
    ax.plot(_epochs(records), vals, color='steelblue')
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set(xlabel='Epoch', ylabel='ACF lag-1',
           title='NAS-08: Residual Temporal Autocorrelation (lag 1)')
    _save(fig, output_path)


def plot_nas12_sv_concentration(records: list[dict], output_path: str):
    """NAS-12: Fraction of variance in top-5 SVs per layer over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = _epochs(records)
    for k in records[0]['signals']['NAS12_sv_concentration']:
        vals = [r['signals']['NAS12_sv_concentration'].get(k) for r in records]
        ax.plot(epochs, vals, label=k)
    ax.set(xlabel='Epoch', ylabel='Top-5 SV fraction',
           title='NAS-12: Singular Value Concentration (1.0 = rank collapse)')
    ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_nas17_layer_redundancy(records: list[dict], output_path: str):
    """NAS-17: Linear CKA between consecutive layers over epochs."""
    if not records[0]['signals']['NAS17_layer_redundancy']:
        return  # single-layer model, nothing to compare
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = _epochs(records)
    for k in records[0]['signals']['NAS17_layer_redundancy']:
        vals = [r['signals']['NAS17_layer_redundancy'].get(k) for r in records]
        ax.plot(epochs, vals, label=k)
    ax.set(xlabel='Epoch', ylabel='Linear CKA',
           title='NAS-17: Layer Redundancy (CKA=1 → layers are identical)')
    ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_nas19_calibration(records: list[dict], output_path: str):
    """NAS-19: Predicted vs target distribution statistics over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = _epochs(records)
    cal = [r['signals']['NAS19_calibration'] for r in records]

    axes[0].plot(epochs, [c['pred_mean'] for c in cal], label='pred')
    axes[0].plot(epochs, [c['target_mean'] for c in cal], linestyle='--', label='target')
    axes[0].set(title='Mean', xlabel='Epoch')
    axes[0].legend()

    axes[1].plot(epochs, [c['pred_std'] for c in cal], label='pred')
    axes[1].plot(epochs, [c['target_std'] for c in cal], linestyle='--', label='target')
    axes[1].set(title='Std Dev', xlabel='Epoch')
    axes[1].legend()

    fig.suptitle('NAS-19: Output Distribution Calibration')
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# plot_all — generate every available plot from a log file
# ---------------------------------------------------------------------------

_REGISTRY = [
    ('NAS01_effective_rank',    plot_nas01_rank_collapse,    'nas01_rank_collapse.png'),
    ('NAS02_gradient_snr',      plot_nas02_gradient_snr,     'nas02_gradient_snr.png'),
    ('NAS04_attn_entropy',      plot_nas04_attn_entropy,     'nas04_attn_entropy.png'),
    ('NAS07_dead_units',        plot_nas07_dead_units,       'nas07_dead_units.png'),
    ('NAS08_residual_acf1',     plot_nas08_acf,              'nas08_residual_acf.png'),
    ('NAS12_sv_concentration',  plot_nas12_sv_concentration, 'nas12_sv_concentration.png'),
    ('NAS17_layer_redundancy',  plot_nas17_layer_redundancy, 'nas17_layer_redundancy.png'),
    ('NAS19_calibration',       plot_nas19_calibration,      'nas19_calibration.png'),
]


def plot_all(log_path: str, output_dir: str):
    """Read JSONL log and save one PNG per available NAS signal."""
    records = load_log(log_path)
    if not records:
        print(f'No records found in {log_path}')
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sig = records[0]['signals']

    for key, fn, fname in _REGISTRY:
        if key in sig:
            fn(records, str(out / fname))
            print(f'  {fname}')

    print(f'Diagnostic plots saved to {output_dir}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python -m visualization.plots <log_path> <output_dir>')
        sys.exit(1)
    plot_all(sys.argv[1], sys.argv[2])
