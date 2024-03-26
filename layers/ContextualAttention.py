import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from layers.MultiHeadAttention import MultiHeadAttention


class ContextualAttention(nn.Module):
    """Contextual Attention as used by Chen et al.

    Given an input of shape [batch_size, seq_len, 2*num_units]
    attention weights are determined for each of the seq_len hidden states.

    The weighted sum of the hidden states of shape [batch_size, 2*num_units] is returned

    Parameters
    ----------
    d_model:
        Length of the input and output vector  (default: 2xgru_dimension)
    attention_dimension:
        Dimension of the output of the layer (default: 2xgru_dimension)
    use_bias:
        Bool specifying whether an bias should be used to retrieve the hidden representation for
        the hidden states of the BiGRU (which serve as values)
    """

    def __init__(self, d_model=24, attention_dimension=24, use_bias=True):
        super().__init__()
        self._use_bias = use_bias

        # The input dimension will be twice the number of units of a single GRU (since it is a BiGRU)
        # This layer calculates a hidden representation of the incoming values for which attention weights are needed
        # It calculates the following: tanh(W h +b)
        self._hidden_rep = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=attention_dimension, bias=self._use_bias),
            nn.Tanh()
        )

        # Apply the attention scoring function (dot-product) between the values and the query vector u
        # Here u is represented as dense layer without bias: u has shape (attention_dimension x 1)
        # Hence, the attention scoring function calculates a dot product as follows: u ° tanh(W h +b)
        self._attn_scoring_fn = nn.Linear(attention_dimension, 1, bias=False)


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
        attention_output = torch.mul(biGRU_outputs, attention_weights).sum(axis=1)

        return attention_output, attention_weights



class MultiHeadContextualAttention(nn.Module):
    """Multihead Contextual Attention

    Given an input of shape [batch_size, seq_len, 2*num_units]
    attention weights are determined for each of the seq_len hidden states.

    The weighted sum of the hidden states of shape [batch_size, 2*num_units] is returned

    Parameters
    ----------
    d_model:
        Length of the input and output vector
    heads:
        Number of heads to use in the MultiHeadAttention
    dropout:
        Dropout ratio to be applied as float. If None, no dropout is applied
    use_reduced_head_dims:
        Bool specifying whether the dimensions of the keys, queries, and values should be reduced
        to avoid large matrices. If true, d_k = d_v = d_q = d_model/h is chosen
    attention_activation_function:
        Can be one of ``'softmax'``, ``'sparsemax'``, ``'entmax15'``, ``'entmax_bisect'``. Default is ``softmax``.
    """

    def __init__(self, d_model, heads, dropout=None,
                 use_reduced_head_dims=False, attention_activation_function="softmax"):
        super().__init__()
        self._use_reduced_head_dims = use_reduced_head_dims

        # Apply the attention scoring function (dot-product) between the values and the query vector u
        # The query needs to be learned
        # Hence, u is represented as trainable tensor, which has shape (d_model x 1)
        u = torch.empty(d_model, 1)
        u = nn.init.xavier_uniform_(u)
        self._query = nn.Parameter(u, requires_grad=True)

        if self._use_reduced_head_dims:
            # To avoid to large matrices, reduce the dimensions of the keys, queries, and values
            # Based on the paper "Attention is all you need" d_k = d_v =d_model/h is chosen
            # In this paper: queries and keys have dimension d_k, i.e., d_q = d_k = d_model/h
            d_k = d_v = d_q = d_model // heads
        else:
            # Since a convolutional reduction block is introduced into MB-M, the matrices are not too large any more
            # after the concatenation of the single lead branch feature maps
            # Hence, using the full dimension for the keys, queries, and values is also an option
            d_k = d_q = d_v = d_model

        self._multihead_attention = MultiHeadAttention(d_model=d_model, k=d_k,
                                                       q=d_q, v=d_v, h=heads,
                                                       dropout=dropout,
                                                       attention_activation_function=attention_activation_function)


    def forward(self, biGRU_outputs):
        # biGRU_outputs is of shape [batch_size, seq_len, 2*num_units], e.g., bs*2250*24
        # Wanted: Seq_len number of attention weights for each element in the batch
        keys = biGRU_outputs
        values = biGRU_outputs

        bs = biGRU_outputs.shape[0]
        # seq_len = biGRU_outputs.shape[1]
        querys = self._query.permute(1, 0)
        querys = querys.repeat(bs, 1, 1)

        attention_output = self._multihead_attention(query=querys, key=keys, value=values)

        # Shape bs x d_model
        return attention_output


if __name__ == "__main__":
    model = MultiHeadContextualAttention(d_model=12 * 2, heads=3)
    summary(model, input_size=(64, 2250, 24), col_names=["input_size", "output_size", "num_params"])

