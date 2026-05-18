# Pre-Training Protocol

## Overview

We use a two-stage training strategy to improve the Switch Transformer backbone:

**Stage 1 — Pre-training**: Train the model on two large public time-series datasets
(M5 Walmart retail sales and UCI Electricity) that share structural similarities with
daily fuel volume forecasting: non-negative counts/consumption, daily granularity,
right-skewed distributions, and meaningful cross-series patterns.

**Stage 2 — Fine-tuning**: Initialize the backbone with the pre-trained weights and
fine-tune on the real gas volume data. Only the backbone layers transfer; the
embedding tables, input projection, and output head are re-initialized to match the
production feature schema.

The hypothesis is that the backbone learns general temporal pattern recognition
(trend, seasonality, regime shifts) from many thousands of series before being
specialized for gas forecasting.

---

## Step 1 — Download Datasets

```bash
bash src/pretrain/download.sh
```

This downloads to `data/pretrain/raw/`:
- `m5-forecasting-accuracy.zip` → extracts `sales_train_validation.csv`, `calendar.csv`
- `electricityloaddiagrams20112014.zip` → extracts `LD2011_2014.txt`

The script is idempotent: already-downloaded files are skipped.

---

## Step 2 — Preprocess

Run from the repository root:

```bash
# M5 Walmart retail sales → data/pretrain/m5_daily.parquet
python src/pretrain/preprocess_m5.py --raw_dir data/pretrain/raw --out_dir data/pretrain

# UCI Electricity → data/pretrain/electricity_daily.parquet
python src/pretrain/preprocess_electricity.py --raw_dir data/pretrain/raw --out_dir data/pretrain
```

Both scripts produce the unified parquet schema (see below).

---

## Step 3 — Pre-train

```bash
python src/train_pretrain.py --output out/pretrain_run1 --max_epochs 50
```

Optional overrides:
```bash
python src/train_pretrain.py \
    --output out/pretrain_run1 \
    --max_epochs 50 \
    --batch_size 128 \
    --data_dir data/pretrain
```

At the end, the script saves:
- `out/pretrain_run1/best_model.pt`     — full model checkpoint
- `out/pretrain_run1/pretrain_backbone.pt` — backbone-only weights for transfer
- `out/pretrain_run1/run_summary.yaml`  — training metrics

---

## Step 4 — Fine-tune with Pre-trained Backbone

```bash
python src/train_switch.py \
    --pretrain_backbone out/pretrain_run1/pretrain_backbone.pt \
    --output out/finetune_run1
```

The `--pretrain_backbone` argument loads backbone weights with `strict=False`,
so the embedding tables, input projection, and output head start from scratch
while the transformer core benefits from pre-training.

---

## Unified Parquet Schema

Both datasets are normalized to this schema before being loaded by the dataloader:

| Column      | Type             | Description                                    |
|-------------|------------------|------------------------------------------------|
| `date`      | datetime64[ns]   | Date of the observation                        |
| `sequence`  | str              | Unique series ID (e.g. `m5__CA_1__FOODS_1__...`) |
| `value`     | float64          | log1p(today's value)                           |
| `value_lag1`| float64          | log1p(yesterday's value)                       |
| `target`    | float64          | log1p(value at t+2, the 2-day-ahead target)    |
| `dataset_id`| int8             | 0=M5, 1=electricity                            |
| `group_id`  | int16            | Coarse group within dataset (see below)        |

**Scaling**: only `np.log1p` applied to raw values — no per-series normalization.
This matches the gas volume distribution (non-negative, right-skewed).

**M5 group_id**: `dept_id` encoded as 0–6 (7 departments: FOODS_1/2/3, HOBBIES_1/2, HOUSEHOLD_1/2).

**Electricity group_id**: clients clustered into 10 deciles by mean daily kWh load using `pd.qcut`.

---

## Adding More Pre-Training Datasets

### Schema to follow

New datasets must produce a parquet file with exactly the columns above.
The `dataset_id` must be a new integer (next available: 2, 3, …).

### Steps

1. **Write a preprocessing script** in `src/pretrain/preprocess_<name>.py`.
   Follow the pattern of `preprocess_m5.py` or `preprocess_electricity.py`:
   - Read raw files
   - Resample to daily if needed
   - Melt to long format with columns `date`, `sequence`, `value`
   - Apply `np.log1p` to `value`
   - Compute `value_lag1 = value.shift(1)` and `target = value.shift(-2)` per sequence
   - Drop NaN rows
   - Assign a new `dataset_id` integer and a meaningful `group_id`
   - Save to `data/pretrain/<name>_daily.parquet`

2. **Register in `dataloader.py`**: in `build_pretrain_dataloaders()`, add a line:
   ```python
   df_new = pd.read_parquet(os.path.join(data_dir, "<name>_daily.parquet"))
   ```
   and include `df_new` in the `pd.concat([...])` call.
   Update `cat_card[0]` (number of unique `dataset_id` values) accordingly —
   currently it is inferred from the data via `df["dataset_id"].nunique()`, so
   adding a new dataset_id will automatically increase it.

3. **Add a download step** to `download.sh`.

### Suggested next datasets

These are structurally similar to daily fuel demand (non-negative, heterogeneous
cross-series patterns) and freely available:

| Dataset               | Series | Freq  | Source                                     |
|-----------------------|--------|-------|--------------------------------------------|
| Monash SF Traffic     | 862    | hourly→daily | https://forecastingdata.org         |
| NYC Citi Bike         | ~800   | daily | https://citibikenyc.com/system-data        |
| PEMS-BAY / PEMS-07    | 325–883| 5-min→daily | https://github.com/liyaguang/DCRNN  |
| Australian Tourism    | 304    | monthly→daily (interpolate) | https://forecastingdata.org |

---

## What Transfers vs What is Re-initialized

### Transferred (backbone weights)
Keys matching `layers.*`, `input_norm*`, `output_norm*`:
- All `SwitchTransformerLayer` modules (self-attention + Switch FFN)
- Input LayerNorm (`input_norm`)
- Output LayerNorm (`output_norm`)

These contain the learned temporal reasoning: attending to relevant history,
routing tokens to appropriate expert FFNs (regime detection), etc.

### Re-initialized (not transferred)
- `embeddings.*` — categorical embedding tables (different categories in production)
- `input_proj.*` — linear projection from raw features to d_model (different n_num/n_cat)
- `linear.*` — output head (different n_target)

Because `load_pretrained_backbone` uses `strict=False`, shape mismatches in
non-backbone layers are silently ignored and those weights start randomly initialized.

### Implications
- If pre-training and fine-tuning use the same `ModelConfig` (same `hidden_dim`,
  `num_layers`, `num_heads`, `ffn_dim`, `num_experts`), the backbone transfers cleanly.
- Changing `hidden_dim` between pre-training and fine-tuning means the backbone
  weights will not match and transfer will fail silently — keep `ModelConfig` consistent.
