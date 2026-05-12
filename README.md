# Auto-Research

This is a simplified research repo. It is aimed for automated research and manual research.

## Setup

We need an environment for local runs and an environment for kubeflow runs.
This is because our kubeflow cluster is running an old version that is clashing with
more recent libraries needed for the models.

    conda env create -f conda/kfp.yaml
    conda env create -f conda/pytorch.yaml
    conda activate pytorch

Create a `.env` file in the repo root with your Kubeflow credentials:

    KUBEFLOW_ENDPOINT=https://dev-kubeflow-1-11.stovell.ai
    KUBEFLOW_USERNAME=<your-email>@stovell.ai
    KUBEFLOW_PASSWORD=<your-password>
    KUBEFLOW_SKIP_TLS_VERIFY=true
    KUBEFLOW_NAMESPACE=main

Create `~/.config/kfp/client.json`:

    {
        "host": "https://dev-kubeflow-1-11.stovell.ai/pipeline"
    }

Authentication uses DEX session cookies (`dex_auth.py`) — no OAuth client credentials needed. The cluster uses a shared `main` namespace.

## Data

The data and dataloaders are fixed and configured by `src/dataloaders.yaml`.

## Running an experiment

Suppose you want to run an experiment named "tcn_bz256".

After modifying `src/train.py` you run the experiment locally or on kubeflow:

Local run:

```bash
python src/train.py --output out/local/{experiment_name}
```

To cap training epochs (e.g. for a quick smoke test):

```bash
python src/train.py --output out/local/{experiment_name} --max_epochs 5
```

Kubeflow run:

```bash
make pipeline.yaml
bash run_kfp.sh {experiment_name}
```

To check the status of the Kubeflow run:

```bash
conda run -n kfp python get_kfp.py --run_id {run_id}
```

If the run completes successfully, the output directory will contain:

- `best_model.pt`
- `curves.png`
- `run_summary.yaml` : the run metrics and training curves data
- `train.log` (kubeflow only)

The format of `run_summary.yaml` is the following:

```
best_val_loss: 0.08469399453939072
best_epoch: 1
curve:
  epoch:
  - 0
  - 1
  - 2
  train_loss:
  - 0.07017532611093201
  - 0.05582243474289632
  - 0.053720542964653464
  val_loss:
  - 0.08631508484748858
  - 0.08469399453939072
  - 0.08501925701940698
  lr:
  - 0.0045000000000000005
  - 0.004050000000000001
  - 0.0036450000000000007
```

If the run summary is not present, the training script crashed, and you can read the log for information.


## Architecture

The pipeline is a standard supervised time-series forecasting loop:

```
dataloaders.yaml  (sidecar to dataloaders.py)
      │
      ▼
dataloaders.py (build_dataloaders)
  ├── columns.py     — typed accessor for column config (date, sequence, categoricals, numericals, targets, scaling)
  ├── encoder.py     — constructs a fit/transform Encoder using composable sklearn-style transformers: Wrapper, GroupScaler, LogScaler, FuncScaler, Encoder
  └── sampler.py     — converts DataFrame → sequence list → SliceDataset (torch Dataset of fixed-length windows)
      │
      ▼
train.py — Model (LSTM + embeddings), training loop, early stopping, writes run_summary.yaml
```

**Key design points:**

- `Columns` (subclass of `DictConfig`) is the single source of truth for which columns play which role. It is passed through the whole pipeline and drives encoder construction, dataloader shape, and model input dimensions.
- `Encoder` in `encoder_lib.py` is a sequential pipeline of transformers (categorical ordinal encoding → per-group StandardScaler on numericals → per-group StandardScaler on targets). `GroupScaler` fits a separate scaler per value of `scaling_column`.
- `SliceDataset` pre-converts the DataFrame into a Python list of sequence dicts (one per `sequence` group), then generates `(start, end)` index pairs on construction. `__getitem__` does the final tensor conversion. This avoids repeated pandas work during training.
- The `Model` in `train.py` embeds categoricals (embedding dim = `floor(log(cardinality)) + 1`), concatenates with numericals, passes through a unidirectional LSTM, and a linear head with softplus output (forces non-negative predictions). Training loss is last-timestep MAE by default; eval loss is always last-timestep MAE.
- `TrainerConfig` and `ModelConfig` are Python dataclasses in `train.py` that define all hyperparameter defaults. Dataloader config lives in `src/dataloaders.yaml`.
- `utils.py` wraps all I/O through `fsspec` (so paths can be local or `gs://...` transparently) and contains shared utilities for config, plotting, and device selection.
