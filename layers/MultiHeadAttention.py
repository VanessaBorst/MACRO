from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, entmax_bisect


class MultiHeadAttention(nn.Module):
    """Multi Head Attention block

    Given 3 inputs of shape (batch_size, T, d_model), that will be used
    to compute query, keys and values, we output an attention
    tensor of shape (batch_size, d_model).

    Parameters
    ----------
    d_model:
        Dimension of the input (and hence, also the output) vector. Here: d_model = 2*num_units
    k:
        Dimension of all key matrices.
    q:
        Dimension of all query matrices.
    v:
        Dimension of all value matrices.
    h:
        Number of heads.
    dropout:
        Dropout ratio to be applied as float. If None, no dropout is applied
    attention_activation_function:
       Can be one of ``'softmax'``, ``'sparsemax'``, ``'entmax15'``, ``'entmax_bisect'``. Default is ``softmax``.
    """

    def __init__(self,
                 d_model: int,
                 k: int,
                 q: int,
                 v: int,
                 h: int,
                 dropout: float,
                 attention_activation_function: str = 'softmax',
                 **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._h = h
        self._k = k  # Needed for the scaled dot product attention
        self._attention_activation_function = attention_activation_function

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q*self._h)

        # Apply tanh after each heads linear key transformation
        self._W_k = nn.Sequential(
            nn.Linear(d_model, k*self._h),
            nn.Tanh()
        )
        self._W_v = nn.Linear(d_model, v*self._h)

        # Output linear function
        self._W_o = nn.Linear(self._h*v, d_model)

        # Score placeholder
        self._scores = None

        # Dropout
        self._dropout = None
        if dropout is not None:
            self._dropout = nn.Dropout(p=dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """Propagate forward the input through the MHA.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated
        and returned with shape (batch_size,d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, 1, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, T, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, T, d_model) used to compute values.

        Returns
        -------
            MH attention tensor with shape (batch_size, d_model).
        """
        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._k)

        # Apply attention scoring function
        match self._attention_activation_function:
            case 'softmax':
                self._scores = F.softmax(self._scores, dim=-1)
            case 'sparsemax':
                self._scores = sparsemax(self._scores, dim=-1)
            case 'entmax15':
                self._scores = entmax15(self._scores, dim=-1)
            case 'entmax_bisect':
                self._scores = entmax_bisect(self._scores, dim=-1)
            case _:
                raise ValueError(f'Attention scoring function {self._attention_activation_function} not supported.')

        if self._dropout is not None:
            self._scores = self._dropout(self._scores)

        attention = torch.bmm(self._scores, values)

        # Concatenate the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention.squeeze()
