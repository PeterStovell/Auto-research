import numpy as np
import pandas as pd

from columns import Columns

# Ordered list of regime feature names — must match what add_regime_features produces.
# Both the pretrain and fine-tune pipelines append these to their numerical columns
# so that n_num is identical and backbone weights transfer correctly.
REGIME_FEATURE_COLS = [
    'roll_std_7',        # 7-day rolling volatility (fast vol regime)
    'roll_std_21',       # 21-day rolling volatility (slow vol regime)
    'trend_7',           # 7-day momentum normalised by 21-day vol
    'roll_spread',       # Roll's implied spread — liquidity / friction proxy
    'roll_autocorr_21',  # 21-day lag-1 autocorrelation — momentum vs mean-reversion regime
    'realized_vol_fwd21',# Forward 21-day realised vol — bid-ask spread teacher signal
                         #   Valid during training (historical data); NaN at sequence tail.
                         #   Set to 0 at true inference time (model handles via imputation task).
]


def add_regime_features(df: pd.DataFrame, value_col: str, seq_col: str) -> list[str]:
    """
    Compute regime-aware features per sequence and append them to df in-place.

    Backward-looking (safe at inference):
      roll_std_7        7-day rolling std of value
      roll_std_21       21-day rolling std of value
      trend_7           (value[t] - value[t-7]) / roll_std_21  (normalised momentum)
      roll_spread       Roll (1984) implied spread:
                          2 * sqrt(max(0, -rolling_lag1_autocov(returns, 21-day)))
                          Proxy for market friction / liquidity.
      roll_autocorr_21  Pearson lag-1 autocorrelation of value over 21-day window.
                          Sign flip = regime boundary (momentum -> mean-reversion).

    Forward-looking teacher signal (NaN for last 21 rows of each sequence):
      realized_vol_fwd21  std(value[t+1..t+21]) — proxy for forward bid-ask spread.
                          Teaches the router to associate current context with the
                          volatility regime that actually materialised.

    Returns REGIME_FEATURE_COLS (list of new column names).
    """
    grp = df.groupby(seq_col, observed=True)

    # ------------------------------------------------------------------ #
    # 1. Rolling volatility                                                #
    # ------------------------------------------------------------------ #
    df['roll_std_7'] = grp[value_col].transform(
        lambda x: x.rolling(7, min_periods=7).std()
    )
    df['roll_std_21'] = grp[value_col].transform(
        lambda x: x.rolling(21, min_periods=21).std()
    )

    # ------------------------------------------------------------------ #
    # 2. Normalised 7-day momentum                                        #
    # ------------------------------------------------------------------ #
    trend_raw = grp[value_col].transform(lambda x: x.diff(7))
    df['trend_7'] = trend_raw / df['roll_std_21'].clip(lower=1e-6)

    # ------------------------------------------------------------------ #
    # 3. Roll's implied spread                                             #
    #    spread = 2 * sqrt(max(0, -Cov(r_t, r_{t-1})))                   #
    #    Cov approx: rolling mean of r_t * r_{t-1} (zero-mean returns)   #
    # ------------------------------------------------------------------ #
    df['_ret'] = grp[value_col].transform(lambda x: x.diff())
    df['_ret_lag'] = grp['_ret'].transform(lambda x: x.shift(1))
    df['_ret_prod'] = df['_ret'] * df['_ret_lag']
    autocov = grp['_ret_prod'].transform(
        lambda x: x.rolling(21, min_periods=14).mean()
    )
    df['roll_spread'] = 2.0 * np.sqrt(np.maximum(0.0, -autocov))
    df.drop(columns=['_ret', '_ret_lag', '_ret_prod'], inplace=True)

    # ------------------------------------------------------------------ #
    # 4. Rolling lag-1 autocorrelation                                    #
    #    pandas rolling().corr() is vectorised                            #
    # ------------------------------------------------------------------ #
    df['roll_autocorr_21'] = grp[value_col].transform(
        lambda x: x.rolling(21, min_periods=14).corr(x.shift(1))
    )

    # ------------------------------------------------------------------ #
    # 5. Forward realised volatility (teacher signal)                     #
    #    f[t] = std(value[t+1..t+21])                                     #
    #    Identity: reverse series -> rolling std -> reverse -> shift(-1)  #
    # ------------------------------------------------------------------ #
    df['realized_vol_fwd21'] = grp[value_col].transform(
        lambda x: (x[::-1].rolling(21, min_periods=21).std()[::-1]).shift(-1)
    )

    return REGIME_FEATURE_COLS


def add_features(df: pd.DataFrame, columns: Columns) -> list[str]:
    """
    Feature engineering for the fine-tune (gas volume) pipeline.
    Called by dataloaders.py after loading the parquet.
    """
    return add_regime_features(df, value_col='NET_QTY', seq_col=columns.sequence())
