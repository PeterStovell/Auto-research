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

## Run training 

Train locally:
```bash
python src/train.py --config configs/puk_monthly_fast.yaml --output out/local/puk_monthly_fast
```

Train on kubeflow:
```bash
make pipeline.yaml
conda run -n kfp python run_kfp.py --config configs/puk_monthly_fast.yaml --output gs://demand-vision/temp/marc/runs/puk_monthly_fast
```
For safety, always use gs://demand-vision/temp/marc as the root for output.

Download kubeflow experiments locally:

```bash
gcloud storage rsync gs://demand-vision/temp/marc/runs out/kfp --recursive
```

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
