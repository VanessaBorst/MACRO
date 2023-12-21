import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from layers.MultiHeadAttention import MultiHeadAttention
from torch.nn import MultiheadAttention as MultiheadAttention_torch

from utils import count_parameters


class ContextualAttention(nn.Module):
    """Contextual Attention as used by Chen et al.

    Given an input of shape [batch_size, seq_len, 2*num_units]
    attention weights are determined for each of the seq_len hidden states.

    The weighted sum of the hidden states of shape [batch_size, 2*num_units] is returned

    Parameters
    ----------
    d_model:
        Length of the input and output vector
    attention_dimension:
        Dimension of the output of the layer (default: 2xgru_dimension)
    use_bias:
        Bool specifying whether an bias should be used to retrieve the hidden representation for
        the hidden states of the BiGRU (which serve as values)
    """

    def __init__(self, gru_dimension=12, attention_dimension=24, use_bias=True):
        super().__init__()
        self._use_bias = use_bias

        # The input dimension will be twice the number of units of a single GRU (since it is a BiGRU)
        # This layer calculates a hidden representation of the incoming values for which attention weights are needed
        # It calculates the following: tanh(W h +b)
        self._hidden_rep = nn.Sequential(
            nn.Linear(in_features=2 * gru_dimension, out_features=attention_dimension, bias=self._use_bias),
            nn.Tanh()
        )

        # Apply the attention scoring function (dot-product) between the values and the query vector u
        # Here u is represented as dense layer without bias: u has shape (attention_dimension x 1)
        # Hence, the attention scoring function calculates a dot product as follows: u ° tanh(W h +b)
        self._attn_scoring_fn = nn.Linear(attention_dimension, 1, bias=False)

        '''OLD:
        # ----- Define the parameters of the layer -----
        # W and b are used to transform the incoming features from the biGRU
        self._W = nn.Parameter(torch.Tensor(in_shape[-1], in_shape[-1]))
        if self._bias:
            self._b = nn.Parameter(toself._hidden_rep = nn.Sequential(
            nn.Linear(in_features=2 * d_model, out_features=attention_dimension, bias=self._use_bias),
            nn.Tanh()
        )

        # Apply the attention scoring function (dot-product) between the values and the query vector u
        # Here u is represented as dense layer without bias: u has shape (attention_dimension x 1)
        # Hence, the attention scoring function calculates a dot product as follows: u ° tanh(W h +b)
        self._attn_scoring_fn = nn.Linear(attention_dimension, 1, bias=False)rch.Tensor(in_shape[-1]))

        # The vector u serves as query of the attention mechanism
        self._u = nn.Parameter(torch.Tensor(in_shape[-1]))

        # ----- Initialize weights with Glorot initialization-----
        nn.init.xavier_uniform(self._W)
        if self._bias:
            nn.init.zeros_(self._b)
        # TODO _u init should be xavier as well
        nn.init.uniform(self._u)
        '''

    def forward(self, biGRU_outputs):
        # biGRU_outputs is of shape [batch_size, seq_len, 2*num_units], e.g., bs*2250*24
        # Wanted: Seq_len number of attention weights for each element in the batch

        hidden_representation = self._hidden_rep(biGRU_outputs)

        # Calculate the score (dot product) between the query and the hidden representations
        # We need to calculate the score for each of the seq_len hidden states of the BiGRU, e.g. seq_len many scores
        # attention_scores has shape [batch_size, seq_len, 1]
        attention_scores = self._attn_scoring_fn(hidden_representation)

        # Remove the 3rd dimension to get shape [batch_size, seq_len]
        # attention_scores = attention_scores.squeeze(2)

        # For each element in the batch, a softmax is applied on each attention_score to get a probability distribution
        # This yields the attention weights of shape [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)

        # Finally, the output of the attention layer is calculated as weighted sum of the BiGRU's hidden states
        # The '*' operator is an element-wise multiplication in Python
        # Alternatively, mul() can be used
        # attention_output = (biGRU_outputs * attention_weights).sum(axis=1)
        attention_output = torch.mul(biGRU_outputs, attention_weights).sum(axis=1)

        return attention_output, attention_weights

        # OLD
        # x = biGRU_outputs
        # u_in = F.tanh(F.add(F.dot(x, self._W), self._b)) if self._bias else F.tanh(F.dot(x, self._W))
        # a = F.softmax(F.dot(u_in, self._u))
        # return x[:, -1, :]  # For testing, just pass through/choose the last hidden state for each element of the batch


class MultiHeadContextualAttention(nn.Module):
    """Multihead Contextual Attention

    Given an input of shape [batch_size, seq_len, 2*num_units]
    attention weights are determined for each of the seq_len hidden states.

    The weighted sum of the hidden states of shape [batch_size, 2*num_units] is returned

    Parameters
    ----------
    d_model:
        Length of the input and output vector
    use_bias:
        Bool specifying whether a bias should be used to retrieve the hidden representation for
        the hidden states of the BiGRU (which serve as values)
    """

    def __init__(self, d_model, heads, dropout=None, use_bias=True, discard_FC_before_MH=False,
                 use_reduced_head_dims=False, use_self_attention=False):
        super().__init__()
        self._use_bias = use_bias
        self._discard_FC_before_MH = discard_FC_before_MH
        self._use_reduced_head_dims = use_reduced_head_dims
        self._use_self_attention = use_self_attention

        if not self._discard_FC_before_MH:
            # The input dimension will be twice the number of units of a single GRU (since it is a BiGRU)
            # This layer calculates a hidden representation of the incoming values for which attention weights are needed
            # It calculates the following: tanh(W h +b)
            self._hidden_rep = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_model, bias=self._use_bias),
                nn.Tanh()
            )

        # Apply the attention scoring function (dot-product) between the values and the query vector u
        if not use_self_attention:
            # The query needs to be learned
            # Hence, u is represented as trainable tensor, which has shape (attention_dimension x 1)
            u = torch.empty(d_model, 1)
            u = nn.init.xavier_uniform_(u)
            self._query = nn.Parameter(u, requires_grad=True)

        if self._use_reduced_head_dims:
            # Previous version:
            # To avoid to large matrices, reduce the dimensions of the keys, queries, and values
            # Based on the paper "Attention is all you need" d_k = d_v =d_model/h is chosen
            # In this paper: queries and keys have dimension d_k, i.e., d_q = d_k = d_model/h
            d_k = d_v = d_q = d_model // heads
        else:
            # Current version (=thesis version):
            # The matrices are not that large anymore, since a convolutional reduction block is introduced
            # after the concatenation of the single lead branch feature maps
            d_k = d_q = d_v = d_model

        self._multihead_attention = MultiHeadAttention(d_model=d_model, k=d_k,
                                                       q=d_q, v=d_v, h=heads,
                                                       dropout=dropout,
                                                       discard_FC_before_MH=self._discard_FC_before_MH)

    def forward(self, biGRU_outputs):
        # biGRU_outputs is of shape [batch_size, seq_len, 2*num_units], e.g., bs*2250*24
        # Wanted: Seq_len number of attention weights for each element in the batch

        keys = self._hidden_rep(biGRU_outputs) if not self._discard_FC_before_MH else biGRU_outputs
        values = biGRU_outputs
        if not self._use_self_attention:
            bs = biGRU_outputs.shape[0]
            # seq_len = biGRU_outputs.shape[1]
            querys = self._query.permute(1, 0)
            querys = querys.repeat(bs, 1, 1)
        else:
            querys = biGRU_outputs    # Self-attention

        attention_output = self._multihead_attention(query=querys, key=keys, value=values)

        # Shape bs x d_model
        return attention_output


class MultiHeadContextualAttentionV2(nn.Module):
    """Multihead Contextual Attention (based on official Torch MHA)

    Given an input of shape [batch_size, seq_len, 2*num_units]
    attention weights are determined for each of the seq_len hidden states.

    The weighted sum of the hidden states of shape [batch_size, 2*num_units] is returned
    """

    def __init__(self, d_model, heads, dropout=None, use_bias=True, discard_FC_before_MH=False,
                 use_mean_query=False):
        super().__init__()
        self._use_bias = use_bias
        self._discard_FC_before_MH = discard_FC_before_MH
        self._use_mean_query = use_mean_query

        if not self._discard_FC_before_MH:
            # The input dimension will be twice the number of units of a single GRU (since it is a BiGRU)
            # This layer calculates a hidden representation of the incoming values for which attention weights are needed
            # It calculates the following: tanh(W h +b)
            self._hidden_rep = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_model, bias=self._use_bias),
                nn.Tanh()
            )

        # Apply the attention scoring function (dot-product) between the values and the query vector u
        if not use_mean_query:
            # The query needs to be learned
            # Hence, u is represented as trainable tensor, which has shape (attention_dimension x 1)
            u = torch.empty(d_model, 1)
            u = nn.init.xavier_uniform_(u)
            self._query = nn.Parameter(u, requires_grad=True)

        d_k = d_v = d_model  # Infers: d_q = d_k = d_model (see Transformer paper)

        self._multihead_attention = MultiheadAttention_torch(embed_dim=d_model, num_heads=heads, dropout=dropout,
                                                             bias=use_bias, kdim=d_k, vdim=d_v, batch_first=True)

    def forward(self, biGRU_outputs):
        # biGRU_outputs is of shape [batch_size, seq_len, 2*num_units], e.g., bs*2250*24
        # Wanted: Seq_len number of attention weights for each element in the batch and weighted sum in the end

        key = self._hidden_rep(biGRU_outputs) if not self._discard_FC_before_MH else biGRU_outputs
        value = biGRU_outputs
        if not self._use_mean_query:
            # The query needs to be learned
            bs = biGRU_outputs.shape[0]
            query = self._query.permute(1, 0)
            query = query.repeat(bs, 1, 1)
        else:
            # Use mean of the BiGRU states as initialization for the query
            query = torch.mean(biGRU_outputs, dim=1)
            query = nn.Parameter(query, requires_grad=True)
            query = query.unsqueeze(1)

        attn_output, attn_output_weights = self._multihead_attention(query, key, value)

        # Shape bs x d_model is required, not bs x 1 x d_model
        attn_output = attn_output.squeeze(1)
        return attn_output


if __name__ == "__main__":
    model = MultiHeadContextualAttentionV2(d_model=12 * 2, heads=3, discard_FC_before_MH=False,
                                           use_mean_query=True)
    summary(model, input_size=(64, 2250, 24), col_names=["input_size", "output_size", "num_params"])
    count_parameters(model)

