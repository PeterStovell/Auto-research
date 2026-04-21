import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_train_metrics(input_path: str, output_dir: str) -> dict:
    df = pd.read_csv(input_path)
    for c in df:
        if c.endswith('_loss'):
            df[c] *= 100  # for readability
    # here mean() picks the only non-nan number
    z = df[['epoch', 'train_loss', 'val_loss', 'test_loss']].groupby('epoch').mean()
    z.insert(2, 'best_val_loss', ((z['val_loss'] == z['val_loss'].min()) * z['val_loss']).replace(0., np.nan))
    z['test_loss'] = ((z['val_loss'] == z['val_loss'].min()) * z['test_loss']).replace(0., np.nan)
    z.plot(style=['-', '--', 'o', 'x'], ylabel='loss')
    plt.savefig(os.path.join(output_dir, 'metrics.png'))
    if 'lr' in df:
        # here mean() picks the only non-nan number
        z = df[['epoch', 'lr']].groupby('epoch').mean()
        z.plot(style=['-'], ylabel='learning rate')
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
    if 'bz' in df:
        # here mean() picks the only non-nan number
        z = df[['epoch', 'bz']].groupby('epoch').mean()
        z.plot(style=['-'], ylabel='batch size')
        plt.savefig(os.path.join(output_dir, 'batch_size.png'))
    # get val_loss and test_loss
    loss_dict = df[['val_loss', 'test_loss']].min().to_dict()
    return loss_dict
