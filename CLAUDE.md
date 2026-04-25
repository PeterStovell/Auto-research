# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Two conda environments are used — one for local training, one for submitting to Kubeflow (they cannot share due to version conflicts):

```bash
conda env create -f conda/pytorch.yaml   # local training
conda env create -f conda/kfp.yaml       # kubeflow pipeline submission
conda activate pytorch
```

Kubeflow credentials go in `~/.config/kfp/client.json` (see README.md for the schema).

## Running Training

Local run:
```bash
cd src && python train.py --output_path ../out/local
```

Config is read from `src/static_config.yaml` (hardcoded path relative to `train.py`). Outputs — model checkpoint, metrics, run summary — are written to `--output_path`.

Run tests:
```bash
cd src && pytest
```

## Kubeflow Deployment

```bash
make image          # build & push Docker image, writes digest to ./image
make pipeline.yaml  # compiles the KFP pipeline
./run.sh configs/puk.yaml   # submit a run
```

`Makefile` chains these: `pipeline.yaml` depends on `train.yaml` which depends on `image`. The compiled `pipeline.yaml` is submitted via `pipeline.py` using the `kfp` (v1) SDK.

To download completed run artifacts locally:
```bash
conda run -n kfp python download.py <experiment_name> <output_path>
```

## Architecture

The pipeline is a standard supervised time-series forecasting loop:

```
static_config.yaml
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
- The Docker image (`FROM pytorch/pytorch`) copies only `src/` and runs `train.py` directly. The Kubeflow component YAML (`train_template.yaml`) has `IMAGE` substituted by `make`.
