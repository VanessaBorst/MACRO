from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).

    Parameters
    ----------
    input dim aka d_model:
        Dimension of the input (and hence, also the output) vector.
    k:
        Dimension of all key matrix.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    dropout:
        Dropout ratio to be applied as float. If None, no dropout is applied
    discard_FC_before_MH:
        If set to True, there is no key transformation before, so the tanh is applied within the MH attention
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    """

    def __init__(self, input_dim: int, embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 kdim=None, vdim=None, **kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        # self.dropout = dropout
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.head_dim = embed_dim // num_heads

        if not self._qkv_same_embed_dim:
            # Query, keys and value matrices
            self.q_proj_weight = nn.Linear(input_dim, embed_dim)
            self.k_proj_weight = nn.Linear(input_dim, self.kdim)
            self.v_proj_weight = nn.Linear(input_dim, self.vdim)
            # self.W_q = nn.Linear(input_dim, q * self._h)
            # self._W_k = nn.Linear(input_dim, k * self._h)
            # self._W_v = nn.Linear(input_dim, v * self._h)
        else:
            # Stack all weight matrices 1...h together for efficiency
            self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)

        # Output linear function
        # self._W_o = nn.Linear(self._h * v, d_model)
        # Todo adapt to the two cases
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        # Score placeholder -> todo delete
        self._scores = None

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        # todo: Eventually, add bias

    def forward(self, x, return_attention=False):
        batch_size, seq_length, _ = x.size()

        if self._qkv_same_embed_dim:
            qkv = self.qkv_proj(x)

            # Separate Q, K, V from linear output
            qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
            q, k, v = qkv.chunk(3, dim=-1)

            # Determine value outputs
            values, attention = scaled_dot_product(q, k, v, mask=mask)
            values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
            values = values.reshape(batch_size, seq_length, self.embed_dim)
            o = self.o_proj(values)
        else:
            #TODO
            pass

        if return_attention:
            return o, attention
        else:
            return o





    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, 1, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, T, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, T, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            MH attention tensor with shape (batch_size, 1, d_model).
        """
        # Old version (works only as long as d_k = d_model):
        # K = key.shape[2]        #shape[1] corresponds to T, shape[2] corresponds to 2x GRU cells
        # After modifications (d_k = d_v = d_model/h in the multi-branch-attention) use self._k instead of K)

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._k)

        # Apply sotfmax
        self._scores = F.softmax(self._scores, dim=-1)

        if self._dropout is not None:
            self._scores = self._dropout(self._scores)

        attention = torch.bmm(self._scores, values)

        # Concatenate the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention.squeeze()
