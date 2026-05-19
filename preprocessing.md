# Preprocessing Notes

## Source Pipeline

The DV preprocessing flow in `~/src/stovell/quark/src/stovell/quark/dv` does not appear to use one central `clean_junk()` function. The cleanup is distributed across the assembly path:

- `dv/assemble.py`
- `dv/dtn_bol/__init__.py`
- `dv/dtn_bol/build_v4.py`
- `dv/build.py`
- `stovell/quark/utils.py`
- `dv/features/monthly_soldto_v3.py`

There is no `dv/train.py` at the inspected path.

## Main Cleanup Steps

The BOL-backed DV models clean and prepare data through these stages:

1. Load raw files through schema-managed readers. Required columns can be dropped via schema fields marked `dropna: true`.
2. If multiple input files are loaded, drop duplicate records with `utils.drop_duplicates(..., keep="last")`.
3. Clip the data to the configured date window using `utils.limit_by_date(...)`. For monthly or weekly aggregation, this can clip to complete calendar periods.
4. Join terminal, rack, product, and DTN-to-OPIS master data.
5. If configured with `dropna` options such as `terminal`, `product`, or `customer`, some joins become inner joins and unmatched rows are removed.
6. Apply config-driven universe filters using `filter_on_universe`, e.g. `COL:VAL1,VAL2` or `COL:~BAD_VALUE`.
7. Load rack/spot price data, dedupe it, forward-fill by product/rack/date keys, and drop rows where forward-filled price values are still missing.
8. Aggregate by configured group level and date period. For volume, `NET_QTY` is summed.
9. Add feature layers. For monthly SPURS-style data, `monthly_soldto_v3.py` creates `y_volume_m0`, `y_volume_m1`, and `y_volume_m2` by shifting monthly `NET_QTY`.

## Implications For auto-research

The current `auto-research` training input appears to be downstream of DV assembly, meaning it is already an assembled parquet rather than raw BOL data. We should avoid recreating the entire DV assembly pipeline inside the dataloader.

Useful cleanup to mirror locally:

- Drop rows missing required identifiers, date, sequence, categoricals, numericals, or targets.
- Drop rows where the training target such as `y_volume_m2` is missing.
- Add config-driven filters for known junk values or unwanted channels/products.
- Drop sequences that cannot produce at least one `seq_len` training or validation window.
- Log row counts before and after each cleanup step.

Be careful with zero volumes. DV code sometimes pads missing volume periods with zero, so zero volume is not automatically junk. Only remove zeros if there is a domain-specific reason.

## Candidate auto-research Cleanup Layer

A good next step is to add a small function in `src/dataloaders.py` before encoding:

```python
def clean_assembled_dataframe(df, columns, cfg):
    # Drop required missing fields.
    required = columns.load_list()
    df = df.dropna(subset=required)

    # Optional config query for project-specific junk filtering.
    if "query" in cfg:
        df = df.query(cfg.query)

    # Optional minimum sequence length filter after date sorting.
    # Keep only sequences with enough rows to form seq_len windows.
    counts = df.groupby(columns.sequence(), observed=True).size()
    valid_sequences = counts.loc[counts >= cfg.seq_len].index
    df = df.loc[df[columns.sequence()].isin(valid_sequences)]

    return df
```

This keeps `auto-research` focused on training-ready parquet cleanup while respecting that heavy DV cleansing already happened upstream.
