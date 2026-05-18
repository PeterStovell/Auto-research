#!/usr/bin/env bash
# Download pre-training datasets to data/pretrain/raw/
# Idempotent: skips files that already exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../../data/pretrain/raw"
mkdir -p "$RAW_DIR"

# ---------------------------------------------------------------------------
# M5 Forecasting Accuracy (from Zenodo record 12636070)
# Contains: sales_train_validation.csv, calendar.csv, sell_prices.csv
# ---------------------------------------------------------------------------
M5_ZIP="$RAW_DIR/m5-forecasting-accuracy.zip"
if [ ! -f "$M5_ZIP" ]; then
    echo "Downloading M5 dataset..."
    curl -L -o "$M5_ZIP" \
        "https://zenodo.org/records/12636070/files/m5-forecasting-accuracy.zip?download=1"
else
    echo "M5 zip already present, skipping."
fi

# Unzip M5 if not already done
if [ ! -f "$RAW_DIR/sales_train_validation.csv" ]; then
    echo "Extracting M5 dataset..."
    unzip -q "$M5_ZIP" -d "$RAW_DIR"
else
    echo "M5 CSV already extracted, skipping."
fi

# ---------------------------------------------------------------------------
# UCI Electricity Load Diagrams 2011-2014
# ---------------------------------------------------------------------------
ELEC_ZIP="$RAW_DIR/electricityloaddiagrams20112014.zip"
if [ ! -f "$ELEC_ZIP" ]; then
    echo "Downloading UCI Electricity dataset..."
    curl -L -o "$ELEC_ZIP" \
        "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
else
    echo "Electricity zip already present, skipping."
fi

if [ ! -f "$RAW_DIR/LD2011_2014.txt" ]; then
    echo "Extracting UCI Electricity dataset..."
    unzip -q "$ELEC_ZIP" -d "$RAW_DIR"
else
    echo "Electricity CSV already extracted, skipping."
fi

echo "All downloads complete. Files in $RAW_DIR:"
ls "$RAW_DIR"
