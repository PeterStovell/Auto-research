import json
import fsspec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from typing import Any


def conf_read(path: str) -> Any:
    with fsspec.open(path, 'r') as f:
        return OmegaConf.load(f)


def conf_write(cfg: Any, path: str):
    with fsspec.open(path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))


def get_accelerator(torch):
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def write_model_architecture(model, metadata_path='/mlpipeline-ui-metadata.json'):
    """Write model architecture as a Kubeflow UI markdown visualization."""
    arch_str = str(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Per-module parameter table (top-level children only)
    rows = [('Module', 'Parameters')]
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        rows.append((name, f'{n:,}'))
    rows.append(('**Total**', f'**{total_params:,}**'))
    rows.append(('**Trainable**', f'**{trainable_params:,}**'))

    table_md = '| ' + ' | '.join(rows[0]) + ' |\n'
    table_md += '|---|---|\n'
    for row in rows[1:]:
        table_md += '| ' + ' | '.join(row) + ' |\n'

    source = f"## Model Architecture\n\n{table_md}\n\n```\n{arch_str}\n```\n"

    metadata = {'outputs': [{'type': 'markdown', 'storage': 'inline', 'source': source}]}
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)


def plot_curves(run_summary, path):
    epochs = run_summary.curve.epoch
    train = run_summary.curve.train_loss
    val = run_summary.curve.val_loss
    best_epoch = run_summary.best_epoch

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train, label="train")
    ax.plot(epochs, val, label="val")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best val (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
