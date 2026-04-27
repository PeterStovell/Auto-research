from typing import Callable

import numpy as np
import pandas as pd
from copy import deepcopy

"""
Helper code to apply scikit-learn transformers to our time series
dataframes with format date, sequence, val1, val2, val3...
"""

class PassThrough:
    def fit(self, df: pd.DataFrame):
        return self
    def transform(self, df: pd.DataFrame, inverse=False):
        return df


class Wrapper:
    """
    Convenience wrapper for a scikit-learn transformer (StandardScaler, MinMaxScaler, etc.)
    """
    def __init__(self, transformer, columns: list[str]):
        self.transformer = transformer
        self.columns = list(columns)

    def fit(self, df: pd.DataFrame):
        self.transformer.fit(df[self.columns])
        return self

    def transform(self, df: pd.DataFrame, inverse=False):
        out = df.copy()
        func = self.transformer.inverse_transform if inverse else self.transformer.transform
        out[self.columns] = func(out[self.columns])
        return out


class GroupScaler:
    """
    Same as wrapper, but by groups.
    """
    def __init__(self, transformer, value_columns: list[str], group_column = 'scaling_column'):
        self.transformer = transformer
        self.value_columns = list(value_columns)
        self.group_column = group_column
        self.scaler = {}

    def fit(self, df: pd.DataFrame):
        for index, group in df.groupby(self.group_column, observed=True):
            self.scaler[index] = deepcopy(self.transformer)
            self.scaler[index].fit(group[self.value_columns])
        return self

    def transform(self, df: pd.DataFrame, inverse=False):
        out = []
        for index, group in df.groupby(self.group_column, observed=True):
            if inverse:
                func = self.scaler[index].inverse_transform
            else:
                func = self.scaler[index].transform
            group[self.value_columns] = func(group[self.value_columns])
            out.append(group)
        out = pd.concat(out, axis=0).sort_index()
        return out


class LogScaler:
    def __init__(self, columns: list[str]):
        self.columns = list(columns)
    def fit(self, df: pd.DataFrame):
        assert set(self.columns).issubset(set(df.columns))
        return self
    def transform(self, df: pd.DataFrame, inverse=False):
        out = df.copy()
        func = (lambda x: np.exp(x) - 1) if inverse else (lambda x: np.log(x + 1))
        out[self.columns] = func(df[self.columns])
        return out


class FuncScaler:
    def __init__(self, columns: list[str], forward: Callable, backward: Callable):
        self.columns = list(columns)
        self.forward = forward
        self.backward = backward
    def fit(self, df: pd.DataFrame):
        assert set(self.columns).issubset(set(df.columns))
        return self
    def transform(self, df: pd.DataFrame, inverse=False):
        out = df.copy()
        func = self.backward if inverse else self.forward
        out[self.columns] = func(df[self.columns])
        return out


class Encoder:
    """
    Final encoders, composed of one transformer for each column group
    """
    def __init__(self, categorical_transformer=None, numerical_transformer=None, target_transformer=None):
        self.target_transformer = target_transformer
        self.transformer_list = [categorical_transformer, numerical_transformer, target_transformer]
        self.transformer_list = [t for t in self.transformer_list if t is not None]

    def fit(self, df: pd.DataFrame):
        for transformer in self.transformer_list:
            transformer.fit(df)
        return self

    def transform(self, df: pd.DataFrame, inverse=False):
        out = df
        for transformer in self.transformer_list:
            out = transformer.transform(out, inverse=inverse)
        return out
