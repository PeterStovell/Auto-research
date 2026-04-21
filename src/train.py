import os
import argparse
import random
import sys
import base64

import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig

from sampler import dataframe_to_sequence_list, pred_to_dataframe, SliceDataset
from sampler import GroupedBatchSampler
from metrics import metric_function_map
from columns import Columns
from plot_utils import plot_train_metrics
from io_utils import pickle_write, json_write, conf_read, conf_write
from torch_utils import get_accelerator


class TorchModel(torch.nn.Module):
    def __init__(self,
                 cat_card: list[int],
                 n_num: int,
                 n_target: int,
                 train_on_all_targets=False,
                 aggregate_loss_lambda=0.,
                 aggregate_scale=1,
                 loss_name=None,
                 loss_config=None,
                 weights=None,
                 ):
        super().__init__()

        if loss_config is None:
            loss_config = {}

        from model import Model
        self.model = Model(cat_card, n_num, n_target)

        if loss_name:
            if loss_name == 'mae':
                self.loss_function = torch.nn.L1Loss(**loss_config)
            elif loss_name == 'mse':
                self.loss_function = torch.nn.MSELoss(**loss_config)
            else:
                raise NotImplementedError(loss_name)
        else:
            print("Reading loss from model file")
            from model import Loss
            self.loss_function = Loss(**loss_config)

        self.train_on_all_targets = train_on_all_targets
        self.aggregate_loss_lambda = aggregate_loss_lambda
        self.aggregate_scale = aggregate_scale
        self.weights = weights

    def forward(self, cat, num):
        return self.model(cat, num)

    def compute_loss(self, batch, device):
        date, seq, cat, num, target = batch
        cat, num, target = cat.to(device), num.to(device), target.to(device)
        pred = self(cat, num)
        if not self.train_on_all_targets:
            pred = pred[:, -1, :]
            target = target[:, -1, :]
        if self.weights is not None:
            pred *= self.weights
            target *= self.weights
        loss = self.loss_function(pred, target)
        if self.aggregate_loss_lambda > 0:
            bz = target.shape[0]
            agg_pred = pred.sum(dim=0)
            agg_target = target.sum(dim=0)
            scale = (1 / bz) ** self.aggregate_scale
            agg_loss = self.loss_function(
                agg_pred * scale, agg_target * scale
            )
            loss = loss + self.aggregate_loss_lambda * agg_loss
        return loss


def run_epoch_train(model, dataloader, max_steps_per_epoch=0, optimizer=None, device='cpu',
                    max_norm=0.,
                    ):
    total_loss = 0.
    n = 0
    model.train()
    for batch in dataloader:
        loss = model.compute_loss(batch, device=device)
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        total_loss += loss.item()
        n += 1
        if n == max_steps_per_epoch:
            break
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
    max_epochs,
    max_steps_per_epoch,
    optimizer_name,
    lr,
    weight_decay,
    gamma,
    max_norm,
    patience,
    device,
    output_path
):

    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    best_val = float("inf")
    patience_left = patience
    best_path = os.path.join(output_path, "best_model.pt")
    metrics = []
    for epoch in range(max_epochs):
        train_loss = run_epoch_train(model, train_loader,
                                     max_steps_per_epoch=max_steps_per_epoch,
                                     optimizer=optimizer, device=device, max_norm=max_norm)
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
            patience_left = patience
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load(best_path))
    return metrics


@torch.no_grad()
def predict(model, dataloader, columns, device, predict_function,
            mc_dropout=0,
            ):
    if mc_dropout > 0:
        model.train()
    else:
        model.eval()
    dfs = []
    if predict_function:
        predict_function = getattr(model.model, predict_function)
        print("predict_function", predict_function)
    for batch in dataloader:
        date, seq, cat, num, target = batch
        cat = cat.to(device)
        num = num.to(device)
        if mc_dropout > 0:
            pred_list = []
            for _ in range(mc_dropout):
                pred = model(cat, num)
                if predict_function:
                    pred = predict_function(pred)
                pred_list.append(pred)
            pred = torch.stack(pred_list, dim=0).mean(0)
        else:
            pred = model(cat, num)
            if predict_function:
                pred = predict_function(pred)
        dfs.append(pred_to_dataframe(date, seq, pred, columns))
    return pd.concat(dfs, axis=0)


def main(cfg: DictConfig):

    print('Loading ', cfg.input_path)
    columns = Columns(cfg['columns'])  # add helper code
    df = pd.read_parquet(cfg.input_path, columns=columns.load_list()).dropna()
    # df = df.sample(frac=0.10)
    if cfg.get('query'):
        print('query:', cfg.query)
        df = df.query(cfg.query).copy()
        print('shape after query:', df.shape)
    print('Adding auxiliary columns')
    columns.add_auxiliary(df)
    print('shape:', df.shape)
    print('nas:')
    print(df.isna().sum())
    print('zeros:')
    print(df[columns.targets()].lt(1).mean())

    print('Encoding and scaling')
    from encoder import get_encoder
    encoder = get_encoder(columns, **cfg.get('encoder_config', {}))
    df_train_val = df.loc[df[columns.date()] < pd.to_datetime(cfg.val_end_date)]
    encoder.fit(df_train_val)
    train_val_sequences = df_train_val[columns.sequence()].unique()
    # keeping only the sequences known at train or val time
    df = df.loc[df[columns.sequence()].isin(train_val_sequences)]
    print('shape after removing unknown sequences:', df.shape)
    df_t = encoder.transform(df)
    print('nas:')
    print(df.isna().sum())
    print('uniques:')
    print(df.nunique())

    print('Transforming dataframe to sequence_list')
    sequence_list = dataframe_to_sequence_list(df_t, columns)

    print("Creating datasets")
    seed = cfg.get('seed', 0)
    random.seed(seed)
    torch.manual_seed(seed)
    train_ds = SliceDataset(sequence_list, length=cfg.seq_len, end_date=cfg.train_end_date)
    print('training samples: ', len(train_ds))
    val_ds = SliceDataset(sequence_list, length=cfg.seq_len, start_date=cfg.train_end_date,
                          end_date=cfg.val_end_date)
    print('validation samples: ', len(val_ds))
    test_ds = SliceDataset(sequence_list, length=cfg.seq_len, start_date=cfg.val_end_date)
    print('test samples: ', len(test_ds))

    print("Creating data loaders")
    if columns.batch_sampling():
        """ Random sampling batches from the same group """
        print(f"batch sampling grouped by {columns.batch_sampling()}")
        sequence2group = df_train_val.groupby('sequence', observed=True)["batch_sampling_column"].agg(
            lambda x: x.value_counts().idxmax()).to_dict()
        group2indices = {}
        for idx in range(len(train_ds)):
            seq = train_ds.get_sequence(idx)['sequence']
            group = sequence2group.get(seq)
            group2indices.setdefault(group, []).append(idx)
        groups = list(group2indices.values())
        batch_sampler = GroupedBatchSampler(groups, batch_size=cfg.batch_size)
        train_dataloader = torch.utils.data.DataLoader(train_ds, batch_sampler=batch_sampler)
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=0,
        )

    val_dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0,
    )

    print("Creating model")
    cat_card = [df[c].nunique() for c in columns.categoricals()]
    n_num = len(columns.numericals())
    n_target = len(columns.targets())
    model_config = OmegaConf.to_container(cfg['model_config']) if 'model_config' in cfg else {}
    loss_config = OmegaConf.to_container(cfg['loss_config']) if 'loss_config' in cfg else {}
    device = cfg.get('accelerator', 'auto')
    if device == 'auto':
        device = get_accelerator()
    weights = torch.tensor(cfg.get('weights'), device=device) if 'weights' in cfg else None
    model = TorchModel(
        cat_card, n_num, n_target,
        cfg.get('train_on_all_targets', False),
        cfg.get('aggregate_loss_lambda', 0,),
        aggregate_scale=cfg.get('aggregate_scale', 1),
        loss_name=cfg.get('loss_name'),
        loss_config=loss_config,
        weights=weights,
    ).to(device)

    os.makedirs(args.output_path, exist_ok=True)

    print('device:', device)

    print("Training model")
    metrics = train_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        max_epochs=cfg.get('max_epochs', 50),
        max_steps_per_epoch=cfg.get('max_steps_per_epoch', -1),
        optimizer_name=cfg.get('optimizer_name', 'Adam'),
        lr=cfg.get('lr', 1e-3),
        weight_decay=cfg.get('weight_decay', 0.),
        gamma=cfg.get('gamma', 1.0),
        max_norm=cfg.get('max_norm', 0.),
        patience=cfg.get('patience', 10),
        device=device,
        output_path=args.output_path,
    )

    print("Testing model")
    test_loss = run_epoch_eval(model, test_dataloader, device=device)
    print(f"test_loss={test_loss:.5f}")

    print("Saving encoder")
    try:
        pickle_write(encoder, os.path.join(args.output_path, 'encoder.pkl'))
    except AttributeError as e:
        print(e)

    print("Predictions and analytics")

    # extract and plot metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df['test_loss'] = test_loss
    metrics_df.to_csv(os.path.join(args.output_path, 'metrics.csv'), index=False)
    loss_dict = plot_train_metrics(os.path.join(args.output_path, 'metrics.csv'), args.output_path)

    # predictions and minimal analytics
    metric_names = cfg.get('metrics', ['wmape'])
    metric_list = []
    for fold, loader in zip(['val', 'test'], [val_dataloader, test_dataloader]):

        metric_list.append({'target': 'all', 'metric': 'loss', 'fold': fold, 'value': loss_dict[f'{fold}_loss']})
        predict_functions = cfg.get("predict_functions", [None])
        for predict_function in predict_functions:
            # generate prediction dataframe: date, sequence, predictions
            pred_df = predict(model, loader, columns, device=device, predict_function=predict_function,
                              mc_dropout=cfg.get('mc_dropout', 0))
            if columns.scaling():
                # join scaling group needed by the target transformer
                pred_df = pd.merge(df[columns.index() + ['scaling_column']], pred_df, on=columns.index())
            # inverse transform the predictions
            if encoder.target_transformer is not None:
                pred_df = encoder.target_transformer.transform(pred_df, inverse=True)
            if columns.scaling():
                pred_df = pred_df.drop('scaling_column', axis=1)
            # join with the actuals, renaming prediction
            pred_df = pd.merge(df[columns.index() + columns.targets()], pred_df,
                               on=columns.index(), suffixes=('', '_pred'))

            output_columns = list(dict.fromkeys(columns.index() + columns.output()))
            extra_df = pd.read_parquet(cfg.input_path, columns=output_columns)
            pred_df = pred_df.merge(extra_df, on=columns.index())
            pred_df.to_parquet(os.path.join(args.output_path, f'{fold}_{predict_function}.parquet'))
            # add metrics
            for target in columns.targets():
                for metric_name in metric_names:
                    if metric_name not in metric_function_map:
                        print(f"Unknown metric name {metric_name}")
                        continue
                    metric_func = metric_function_map[metric_name]
                    score = metric_func(pred_df[target], pred_df[target + '_pred'])
                    print(target, metric_name, fold, predict_function, score)
                    metric_list.append({'target': target, 'metric': metric_name, 'fold': fold,
                                        'predict_function': predict_function,
                                        'value': score})

            if predict_function == 'predict_9' and hasattr(model.model, 'sort_models'):
                # here we sort individual models so the ensemble predict functions
                # can use the top models
                # TODO: make the code less hacky
                getattr(model.model, 'sort_models')(metric_list)

    metric_df = pd.DataFrame(metric_list)
    metric_df.to_csv(os.path.join(args.output_path, 'all_metrics_long.csv'), index=False)
    metric_dfp = metric_df.pivot(index=['target', 'metric', 'predict_function'], columns='fold', values='value').reset_index()

    # save metrics and arguments
    metric_dfp.to_csv(os.path.join(args.output_path, 'all_metrics.csv'), index=False)
    json_write(vars(args), os.path.join(args.output_path, 'args.json'))
    conf_write(cfg, os.path.join(args.output_path, 'cfg.yaml'))
    print(f'results in {args.output_path}')

    if args.kubeflow_metrics_path:
        summary_dict = metric_df.query("metric!='loss'").groupby('fold')['value'].mean().to_dict()
        dirname = os.path.dirname(args.kubeflow_metrics_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        metrics = {'metrics': [{'name': k, 'numberValue':  v} for k, v in summary_dict.items()]}
        json_write(metrics, args.kubeflow_metrics_path)

    if args.kubeflow_metadata_path:
        image_path = os.path.join(args.output_path, 'metrics.png')
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        header = metric_dfp.columns.values.tolist()
        tmp_path = os.path.join(args.output_path, 'all_metrics_noheader.csv')
        metric_dfp.to_csv(tmp_path, index=False, header=False, float_format='%.3f')
        data = open(tmp_path, 'r').read()
        metadata = {"outputs": [
            {"type": "web-app", "storage": "inline", "source": f"<img src='data:image/png;base64,{img_b64}'>"},
            {"type": "table", "format": "csv", "storage": "inline", "header": header, "source": data},
        ]}
        dirname = os.path.dirname(args.kubeflow_metadata_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        json_write(metadata, args.kubeflow_metadata_path)


if __name__ == '__main__':
    print("python", sys.version)
    print("pytorch", torch.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default=f'./out', help='output path')
    parser.add_argument('--kubeflow_metrics_path', default=None, help='kf metrics path')
    parser.add_argument('--kubeflow_metadata_path', default=None, help='kf metadata path')
    parser.add_argument('--max_epochs', type=int, default=-1)
    parser.add_argument('--max_steps_per_epoch', type=int, default=-1)
    args = parser.parse_args()
    config_path = os.path.realpath(__file__).replace("train.py", "config.yaml")
    cfg = conf_read(config_path)
    main(cfg)
