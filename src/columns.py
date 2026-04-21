from omegaconf import DictConfig
import pandas as pd


"""
Helper class for dataframe columns
"""

def concat(df):
    if len(df.columns) == 1:
        return df[df.columns[0]]
    return df.agg('-'.join, axis=1)


class Columns(DictConfig):
    def date(self):
        return self.get('date', 'date')

    def sequence(self):
        return self.get('sequence', 'sequence')

    def categoricals(self):
        return self['categoricals']

    def numericals(self):
        return list(self['numericals'])

    def targets(self):
        return list(self['targets'])

    def scaling(self):
        return self.get('scaling', [])

    def batch_sampling(self):
        return self.get('batch_sampling', [])

    def output(self):
        return self.get('output', [])

    def add_auxiliary(self, df: pd.DataFrame):
        """
        auxiliary columns are concatenated categoricals,
        added to make data manipulation easier (like the sequence column)
        """
        if self.scaling():
            df[f'scaling_column'] = concat(df[self.scaling()])
        if self.batch_sampling():
            df[f'batch_sampling_column'] = concat(df[self.batch_sampling()])

    def index(self):
        return [self["date"], self["sequence"]]

    def encoder_list(self):
        return self.categoricals()

    def load_list(self):
        all_cols = self.index() + self.categoricals() \
                + self.scaling() + self.batch_sampling() \
                + self.numericals() + self.targets()
        return list(dict.fromkeys(all_cols))
