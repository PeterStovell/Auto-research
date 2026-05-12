# Dataset Column Reference

**File:** `~/src/stovell/data/dv/dataframe.parquet`  
**Shape:** 861,818 rows × 130 columns | Date range: 2024-01-01 – 2026-03-16

---

## Column Groups

| Column(s) | Type | Description | Recommendation |
|---|---|---|---|
| `date` | date | Daily timestamp. Extract `day_of_week`, `is_weekend`, `week_of_year` for temporal signals. | Derive calendar features |
| `sequence` | sequence id | Unique time-series identifier (1,085 unique). Defines the granularity of each forecasting series. | Required — index key |
| `Channel ID` | categorical (4) | Distribution channel: BRND (branded retail), UNBS (unbranded), UNBC (unbranded commercial), PS&D. UNBS has 67% zero target rate. | Use as categorical + scaling column |
| `GRADE_NAME` | categorical (6) | Fuel grade: NO2 (diesel), PRE (premium), NO1, HOI, KER, etc. NO1/HOI/KER are >66% zero. | Use as categorical |
| `RACK_STATE` | categorical (40) | US state of the rack terminal. Several states (RI, SD, ND, MN, IL) are 91–99% zeros — near-inactive markets. | Use as categorical; consider filtering dead states |
| `brand_indicator` | categorical (2) | Branded (b) vs unbranded (u) fuel. Unbranded has higher zero rate (54% vs 41%). | Use as categorical |
| `RACK_ID` | categorical (204) | Unique rack/terminal identifier. Several IDs (e.g., 158, 161, 80, 84) are 99.75% zeros. | Use as categorical; filter near-zero racks |
| `RACK_NAME` | identifier | Human-readable name for `RACK_ID`. One-to-one with RACK_ID. | Drop — redundant with RACK_ID |
| `TERM_NAME` | identifier | Terminal name (60 unique). Closely related to `RACK_ID`. | Drop — redundant with RACK_ID |
| `TERM_TIME_ZONE` | categorical (5) | Terminal timezone (America/Chicago, America/Denver, etc.). Proxy for geographic region. 5 unique values. | Optional categorical |
| `PCLASS_NAME` | categorical | Product class name (fuel product grouping). Related to GRADE_NAME. | Check cardinality; may be redundant with GRADE_NAME |
| `octane_level` | numerical | Octane rating of the fuel product (0, 85, 86, 87, 91, 92, 93). 7 distinct values — behaves as ordinal. | Use as categorical or ordinal numerical |
| `spot_market` | categorical (7) | Spot market region the rack is priced against (e.g., Gulf Coast, Group 3, Chicago). 7 unique regions. | Use as categorical |
| `level` | — | Constant zero across all rows. No variance. | **Drop** |
| `supplier` | — | Constant `Valero` (single supplier dataset). No variance. | **Drop** |
| `opis_rack_name` | identifier | OPIS-specific rack name (208 unique). Alternative identifier. | Drop — redundant |
| `rack_product_name` | identifier | Product name at the rack level (14 unique). | Drop — redundant with GRADE_NAME |
| `spot_product_name` | identifier | Product name in the spot market (11 unique). | Drop — redundant with GRADE_NAME |
| `NET_QTY` | numerical (target-input) | Daily delivered volume in gallons. Primary demand signal. Mean: 48k, std: 295k (very skewed). 88% of target zeros co-occur with NET_QTY=0. | Use as numerical input |
| `y_volume_t1` | numerical (lag target) | Realized volume at t+1 (1-day ahead). Strong autoregressive signal for predicting t+2. ~0.8% nulls. | Use as numerical input |
| `y_volume_t2` | numerical (**target**) | Realized volume at t+2 (2-day ahead). **Current training target.** ~1.1% nulls, ~48% zeros. | Target |
| `y_volume_t2t3` | numerical (lag target) | Cumulative volume over t+2 to t+3. ~1.4% nulls. | Optional numerical |
| `y_volume_t2t4` | numerical (lag target) | Cumulative volume over t+2 to t+4. ~1.6% nulls. | Optional numerical |
| `y_volume_t2t5` | numerical (lag target) | Cumulative volume over t+2 to t+5. ~1.8% nulls. | Optional numerical |
| `y_volume_t2t6` | numerical (lag target) | Cumulative volume over t+2 to t+6. ~2.0% nulls. | Optional numerical |
| `volume_2D` … `volume_5D` | numerical (rolling) | Rolling sum of NET_QTY over 2–5 day windows. Short-range momentum signal. | Use as numerical |
| `NET_QTY_mean{1,7,14,28,84,168}D_{group}` | numerical (rolling, 32 cols) | Rolling means of NET_QTY over 1–168 days, grouped by RACK_ID, GRADE_NAME+RACK_PADD, brand_indicator+RACK_PADD, and GRADE_NAME+brand_indicator+RACK_ID. Comprehensive multi-scale demand baseline. | Use as numerical |
| `NET_QTY_std{1,7,14,28,84,168}D_{group}` | numerical (rolling, 32 cols) | Rolling standard deviations of NET_QTY — same groupings as means above. Measures demand volatility across time scales. | Use as numerical |
| `NET_QTY_diff_ema_{1,7,14,21}` | numerical (EMA, 4 cols) | Exponential moving average of NET_QTY day-over-day difference. Captures short/medium-term demand trend. No nulls. | Use as numerical |
| `NET_QTY_diff_abs_ema_{1,7,14,21}` | numerical (EMA, 4 cols) | EMA of the absolute day-over-day change in NET_QTY. Captures demand volatility magnitude regardless of direction. No nulls. | Use as numerical |
| `hier_{1..6}_7D_meansum` | numerical (hierarchical, 6 cols) | 7-day rolling mean-sum of NET_QTY at hierarchical aggregation levels 1–6. Captures demand from the broader distribution network. | Use as numerical |
| `hier_{1..6}_7D_std` | numerical (hierarchical, 6 cols) | 7-day rolling standard deviation of NET_QTY at hierarchy levels 1–6. | Use as numerical |
| `long_7D_168D_mean` | numerical (long-range) | Ratio or difference between 7-day and 168-day (24-week) rolling means. Long-range trend signal. ~0.9% nulls. | Use as numerical |
| `long_7D_168D_std` | numerical (long-range) | Ratio or difference between 7-day and 168-day rolling standard deviations. Long-range volatility signal. ~1.1% nulls. | Use as numerical |
| `cwt_morl_laststd_NET_QTY_{1..7}` | numerical (wavelet, 7 cols) | Continuous wavelet transform (Morlet) coefficients of NET_QTY, standardized by last period. Captures cyclical/frequency-domain patterns. | Use as numerical |
| `smart_volume_t2_{low,mid,high}` | numerical (forecast, 3 cols) | External model forecasts for t+2 volume at low/mid/high confidence bounds. Strong prior signal. | Use as numerical |
| `smart_volume_t2t3_{low,mid,high}` | numerical (forecast, 3 cols) | External model forecasts for cumulative t+2–t+3 volume. | Use as numerical |
| `smart_volume_t2t4_{low,mid,high}` | numerical (forecast, 3 cols) | External model forecasts for cumulative t+2–t+4 volume. | Use as numerical |
| `smart_volume_t2t5_{low,mid,high}` | numerical (forecast, 3 cols) | External model forecasts for cumulative t+2–t+5 volume. | Use as numerical |
| `smart_volume_t2t6_{low,mid,high}` | numerical (forecast, 3 cols) | External model forecasts for cumulative t+2–t+6 volume. | Use as numerical |
| `rack_gross_price` | numerical (price) | Gross rack price of fuel ($/gallon). 24.6% nulls. | Use with imputation |
| `branded_average_price` | numerical (price) | Average branded retail fuel price. 9.4% nulls. | Use with imputation |
| `unbranded_low_price` | numerical (price) | Lowest unbranded wholesale price. 9.2% nulls. | Use with imputation |
| `spot_price` | numerical (price) | Spot market reference price. 33.7% nulls — highest missing rate. | Use cautiously or drop |
| `agg_branded_average_price_diff1D_{group}` | numerical (price diff) | 1-day change in branded average price, aggregated by GRADE_NAME+RACK_ID+brand_indicator. | Use as numerical |
| `agg_rack_gross_price_diff1D_{group}` | numerical (price diff) | 1-day change in rack gross price, by group. | Use as numerical |
| `agg_spot_price_diff1D_{group}` | numerical (price diff) | 1-day change in spot price, by group. | Use as numerical |
| `agg_unbranded_low_price_diff1D_{group}` | numerical (price diff) | 1-day change in unbranded low price, by group. | Use as numerical |
| `agg_NET_QTY_diff1D_{group}` | numerical (volume diff) | 1-day change in NET_QTY aggregated by group. Immediate demand momentum. | Use as numerical |
| `agg_NET_QTY_diff7D_{group}` | numerical (volume diff) | 7-day change in NET_QTY by group. Medium-term demand shift. | Use as numerical |
| `RACK_PADD` | categorical | PADD region of the rack (Petroleum Administration for Defense Districts). Geographic demand zone. | Use as categorical |

---

## Zero Analysis Summary (`y_volume_t2`)

| Cause | Zero Rate | Notes |
|---|---|---|
| `Channel ID = UNBS` | 67% | Unbranded supply — structural low-activity channel |
| `GRADE_NAME = NO1` | 92% | Heating oil #1 — rare fuel type |
| `GRADE_NAME = HOI` | 76% | High-octane intermediate — limited distribution |
| Dead rack states (RI, SD, ND, MN, IL) | 91–99% | Near-inactive markets |
| Dead RACK_IDs (e.g., 158, 161) | 99.75% | Inactive terminals |
| Weekend / non-delivery days | ~29% (2/7) | Contributes but not the dominant driver |
| NET_QTY also zero | 88% of zeros | Most zeros are input-driven (supply-side) |

**Bottom line:** The 48% zero rate is dominated by structural inactivity in specific channels, grades, and racks — not just calendar effects. Filtering or stratifying by `Channel ID` and `GRADE_NAME` would significantly reduce noise.
