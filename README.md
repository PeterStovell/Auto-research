# Auto-Research

This is a simplified research repo. It is aimed for automated research and manual research.

## Setup

We need an environment for local runs and an environment for kubeflow runs.
This is because our kubeflow cluster is running an old version that is clashing with
more recent libraries needed for the models.

    conda env create -f conda/kfp.yml
    conda env create -f conda/pytorch.yml
    conda activate pytorch

The kfp client configuration must be located here:

    ~/.config/kfp/client.json

example content:

    {
        "host": "https://kubeflow18.endpoints.dev-stovell-ai.cloud.goog/pipeline", 
        "client_id": "811609456607-61mmvhq0o7vq4rgu25hkl25g7bulug1t.apps.googleusercontent.com", 
        "other_client_id": "811609456607-i25dvmvcr9ousd48mp03fsrnhsq0ajdc.apps.googleusercontent.com", 
        "other_client_secret": "...", 
        "namespace": "marc"
    }


## Running an experiment

Suppose you want to run the experiment named "tcn_bz256" on the config named "puk_monthly".

After modifying the code and config you run the experiment locally or on kubeflow:

Local experiment:

```bash
python src/train.py --config configs/{config_name}.yaml --output out/local/{config_name}/{experiment_name}
```

Kubeflow experiment:

```bash
make pipeline.yaml
conda run -n kfp python run_kfp.py --config configs/{config_name}.yaml --output gs://demand-vision/temp/marc/runs/{config_name}/{experiment_name}
# wait for job completion
gcloud storage rsync gs://demand-vision/temp/marc/runs out/kfp --recursive
```

If the run completes successfully, the output directory will contain:

- `best_model.pt`
- `curves.png`
- `run_summary.yaml` : the run metrics and training curves data
- `train.log` (kfp only)

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
config.yaml
      │
      ▼
dataloaders.py (build_dataloaders)
  ├── columns.py    — typed accessor for column config (date, sequence, categoricals, numericals, targets, scaling)
  ├── encoder.py    — constructs a fit/transform Encoder from encoder_lib primitives
  ├── encoder_lib.py — composable sklearn-style transformers: Wrapper, GroupScaler, LogScaler, FuncScaler, Encoder
  └── sampler.py    — converts DataFrame → sequence list → SliceDataset (torch Dataset of fixed-length windows)
      │
      ▼
train.py — Model (LSTM + embeddings), training loop, early stopping, writes run_summary.yaml
```

**Key design points:**

- `Columns` (subclass of `DictConfig`) is the single source of truth for which columns play which role. It is passed through the whole pipeline and drives encoder construction, dataloader shape, and model input dimensions.
- `Encoder` in `encoder_lib.py` is a sequential pipeline of transformers (categorical ordinal encoding → per-group StandardScaler on numericals → per-group StandardScaler on targets). `GroupScaler` fits a separate scaler per value of `scaling_column`.
- `SliceDataset` pre-converts the DataFrame into a Python list of sequence dicts (one per `sequence` group), then generates `(start, end)` index pairs on construction. `__getitem__` does the final tensor conversion. This avoids repeated pandas work during training.
- The `Model` in `train.py` embeds categoricals (embedding dim = `floor(log(cardinality)) + 1`), concatenates with numericals, passes through a unidirectional LSTM, then a LayerNorm and linear head with softplus output (forces non-negative predictions). Training loss uses sequence-weighted MAE (later timesteps weighted higher); eval loss uses last-timestep MAE only.
- Config is managed via OmegaConf. `DataLoaderConfig`, `TrainerConfig`, and `ModelConfig` are Python dataclasses used as defaults and merged with the YAML config.
- `io_utils.py` wraps all I/O through `fsspec`, so paths can be local or GCS (`gs://...`) transparently.
