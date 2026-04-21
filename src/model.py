import numpy as np
import torch
import torch.nn as nn


class Model(torch.nn.Module):

    def __init__(self, cat_card:list[int], n_num:int, n_target:int,
                 hidden_dim:int=64,
                 num_layers:int=2,
                 dropout:float=0.25,
                 ):
        super().__init__()
        emb_dims = [int(np.log(card) + 1) for card in cat_card]
        input_dim = sum(emb_dims) + n_num
        self.embeddings = nn.ModuleList([nn.Embedding(card, emb_dim)
                                         for card, emb_dim in zip(cat_card, emb_dims)])
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )
        self.linear = nn.Linear(in_features=hidden_dim, out_features=n_target)

    def forward(self, cat, num):
        """
        cat: tensor of indices, shape = [batch_size, seq_len, n_cat] with n_cat = len(cat_card)
        num: tensor of floats, shape = [batch_size, seq_len, n_num]

        Note that the batch dimension is optional
        """

        # embedding of categories
        x = [emb(cat[..., i]) for i, emb in enumerate(self.embeddings)]
        # concatenate embeddings and numerical values along last axis
        x = torch.cat(x + [num], dim=-1)  # shape is [batch_size, seq_len, input_dim]
        y, _ = self.lstm(x)  # shape is [batch_size, seq_len, hidden_dim]
        y = self.linear(y)  # shape is [batch_size, seq_len, n_target]
        y = nn.functional.softplus(y)
        return y
