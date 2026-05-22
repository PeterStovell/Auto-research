import random
from datetime import datetime
import pandas as pd
import torch
from columns import Columns

"""
Utilities to feed data sample to the models.

Terminology for this module:

 - dataframe: our usual pandas dataframe indexed by date_column and sequence_column, 
        containing features and targets for all sequences
 - sequence: a dictionary representing a full sequence in plain python (dates, features, targets)
 - slice: a tuple representing a subsequence ready to be fed to a torch models for training and inference
 
We convert the input dataframe into a list of sequences. 
The list of sequences is then used by a torch data loader to create slices on the fly.
"""


def dataframe_to_sequence_list(df: pd.DataFrame, columns: Columns) -> list[dict]:
    """
    Converts a dataframe into a list of sequences.
    The list of sequences facilitates the data access for the torch dataset.
    """
    sequence_list = []
    for sequence_index, sequence_df in df.sort_values(columns.date()).groupby(
            columns.sequence(), observed=True):
        sequence = {
            'date': sequence_df[columns.date()].to_list(),
            'sequence': sequence_index,
            'categorical': sequence_df[columns.categoricals()].values.tolist(),
            'numerical': sequence_df[columns.numericals()].values.tolist(),
            'target': sequence_df[columns.targets()].values.tolist(),
        }
        sequence_list.append(sequence)
    return sequence_list


def sequence_to_slice(sequence: dict, start: int, end: int) -> tuple:
    """
    Extract a slice from a sequence.
    slices are produced in torch format so they can be fed directly to a models for training and inference.
    """
    date = str(sequence['date'][end - 1])  # string format because date formats are not allowed
    seq = sequence['sequence']
    cat = torch.tensor(sequence['categorical'][start:end], dtype=torch.long)
    num = torch.tensor(sequence['numerical'][start:end], dtype=torch.float32)
    target = torch.tensor(sequence['target'][start:end], dtype=torch.float32)
    return date, seq, cat, num, target


def get_slice_indexes(sequence_list:list[dict], length, start_date=None, end_date=None):
    """
    Extract from a list of sequences all the valid slice indexes given the desired length and dates.
    """
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    if end_date is not None:
        end_date = pd.to_datetime(end_date)

    slice_indexes = []
    for i in range(0, len(sequence_list)):
        dates = sequence_list[i]['date']
        for j in range(0, len(dates) - length):
            last_date = pd.to_datetime(dates[j + length - 1])
            if (start_date is None or last_date >= start_date) and (end_date is None or last_date < end_date):
                slice_indexes.append((i, j, j + length))
    return slice_indexes


class SliceDataset(torch.utils.data.Dataset):
    """
    Constructed from a list of sequence, this torch Dataset will generate all the valid slices
    given the desired length and dates.
    """
    def __init__(self, sequence_list: list[dict], length=42, start_date=None, end_date=None):
        self.sequence_list = sequence_list
        self.length = length
        self.slice_indexes = get_slice_indexes(sequence_list, length, start_date, end_date)

    def __len__(self):
        return len(self.slice_indexes)

    def __getitem__(self, idx):
        i, start, end = self.slice_indexes[idx]
        sequence = self.sequence_list[i]
        return sequence_to_slice(sequence, start, end)

    def get_sequence(self, idx):
        i, _, _ = self.slice_indexes[idx]
        return self.sequence_list[i]


class GroupedBatchSampler(torch.utils.data.Sampler):

    def __init__(self, groups:list[list[int]], batch_size: int=32):
        super().__init__()
        self.groups = groups
        self.batch_size = batch_size

    def __iter__(self):
        all_batches = []
        for group in self.groups:
            random.shuffle(group)
            for i in range(0, len(group), self.batch_size):
                all_batches.append(group[i:i + self.batch_size])
        random.shuffle(all_batches)
        yield from all_batches

    # def __len__(self):
    #     return len(self.groups)


def pred_to_dataframe(date: str, seq:str, pred: torch.Tensor, columns: Columns) -> pd.DataFrame:
    """
    Converts torch predictions back to a dataframe.
    """
    index_df = pd.DataFrame({
            columns.date(): [datetime.fromisoformat(d) for d in date],
            columns.sequence(): seq,
        })
    pred_df = pd.DataFrame(pred[:, -1, :].cpu().detach().numpy(), columns=columns.targets())
    out = pd.concat([index_df, pred_df], axis=1)
    return out
