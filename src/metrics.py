import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def wmape(actuals: pd.Series, predictions:pd.Series) -> float:
    return 100 * mean_absolute_error(actuals, predictions) / mean_absolute_error(actuals, predictions*0)


def non_zero_wmape(actuals: pd.Series, predictions:pd.Series) -> float:
    sample_weight = actuals.gt(0)
    return (100 * mean_absolute_error(actuals, predictions, sample_weight=sample_weight) /
            mean_absolute_error(actuals, predictions*0, sample_weight=sample_weight))


def accuracy(actuals: pd.Series, predictions:pd.Series):
    return 100 * np.mean((actuals > 0) == (predictions > 0))


metric_function_map = {
    'wmape': wmape,
    'non_zero_wmape': non_zero_wmape,
    'accuracy': accuracy,
}
