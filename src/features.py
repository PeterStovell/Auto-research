import pandas as pd

from columns import Columns


def add_features(df: pd.DataFrame, columns: Columns) -> list[str]:
    """No-op feature engineering. Returns an empty list of new columns."""
    return []
