import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_curves(run_summary, path):
    epochs = run_summary.curve.epoch
    train = run_summary.curve.train_loss
    val = run_summary.curve.val_loss
    best_epoch = run_summary.min_epoch

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
