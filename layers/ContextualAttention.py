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
    gru_dimension:
        Number of units contained per GRU
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
            nn.Linear(in_features=2 * gru_dimension, out_features=attention_dimension, bias=self._use_bias),
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
    gru_dimension:
        Number of units contained per GRU
    attention_dimension:
        Dimension of the output of the layer (default: 2xgru_dimension)
    use_bias:
        Bool specifying whether an bias should be used to retrieve the hidden representation for
        the hidden states of the BiGRU (which serve as values)
    """
    def __init__(self, gru_dimension, heads, dropout=None, use_bias=True):
        super().__init__()
        self._use_bias = use_bias
        d_model = 2 * gru_dimension

        # The input dimension will be twice the number of units of a single GRU (since it is a BiGRU)
        # This layer calculates a hidden representation of the incoming values for which attention weights are needed
        # It calculates the following: tanh(W h +b)
        self._hidden_rep = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model, bias=self._use_bias),
            nn.Tanh()
        )

        # Apply the attention scoring function (dot-product) between the values and the query vector u
        # Here u is represented as trainable tensor, which has shape (attention_dimension x 1)
        u = torch.empty(d_model, 1)
        u = nn.init.xavier_uniform_(u, gain=nn.init.calculate_gain('linear'))
        self._query = nn.Parameter(u, requires_grad=True)

        self._multihead_attention = MultiHeadAttention(d_model=d_model, k=d_model,
                                                       q=d_model, v=d_model, h=heads,
                                                       dropout=dropout)

    def forward(self, biGRU_outputs):
        # biGRU_outputs is of shape [batch_size, seq_len, 2*num_units], e.g., bs*2250*24
        # Wanted: Seq_len number of attention weights for each element in the batch

        keys = self._hidden_rep(biGRU_outputs)
        values = biGRU_outputs

        bs = biGRU_outputs.shape[0]
        # seq_len = biGRU_outputs.shape[1]
        # querys = self._query.repeat(seq_len, 1, 1).repeat(bs, 1, 1, 1).squeeze()
        querys = self._query.permute(1, 0)
        querys = querys.repeat(bs, 1, 1)

        attention_output = self._multihead_attention(query=querys, key=keys, value=values)

        # Shape bs x 24
        return attention_output


if __name__ == "__main__":
    model = MultiHeadContextualAttention()
    summary(model, input_size=(2, 2250, 24), col_names=["input_size", "output_size", "num_params"])
