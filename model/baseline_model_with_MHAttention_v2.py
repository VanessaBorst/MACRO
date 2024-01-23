import torch.nn as nn
from torchinfo import summary

from base import BaseModel
from layers.ContextualAttention import ContextualAttention, MultiHeadContextualAttention, \
    MultiHeadContextualAttentionV2, MultiHeadContextualAttentionV3, MultiHeadContextualAttentionV4, \
    MultiHeadContextualAttentionV5
from utils import plot_record_from_np_array


class BaselineModelWithMHAttentionV2(BaseModel):
    def __init__(self, apply_final_activation, multi_label_training, gru_units=12, dropout_attention=0.2, heads=3,
                 discard_FC_before_MH=False, num_classes=9,
                 use_reduced_head_dims=None,
                 attention_activation_function=None,
                 attention_special_init=None,
                 attention_type="v2"):
        """
        :param apply_final_activation: whether the Sigmoid(sl) or the LogSoftmax(ml) should be applied at the end
        :param multi_label_training: if true, Sigmoid is used as final activation, else the LogSoftmax
        :param num_classes: Num of classes to classify
        :param num_cnn_blocks: Num of CNN blocks to use
        """
        super().__init__()

        assert use_reduced_head_dims is None or attention_type == "v1", \
            "use_reduced_head_dims can only be used with attention_type=v1!"

        assert attention_activation_function is None or attention_type == "v1", \
            "attention_activation_function can only be used with attention_type=v1!"

        assert attention_special_init is None or attention_type == "v1", \
            "attention_special_init can only be used with attention_type=v1!"

        if use_reduced_head_dims:
            assert (2 * gru_units) % heads == 0, \
                "Twice the number of GRU cells (d_model) must be divisible by num_heads!"

        self._apply_final_activation = apply_final_activation
        self._conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=24, stride=2, padding=11),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )
        self._conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=24, stride=2, padding=11),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )
        self._conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=24, stride=2, padding=11),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )
        self._conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=24, stride=2, padding=11),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )
        # Last block has a convolutional pooling with kernel size 48!
        self._conv_block5 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=48, stride=2, padding=23),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )

        # Without last option the input would have to be (seq_len, batch, input_size)
        # With batch_first it can be of the shape (batch, seq_len, input/feature_size)
        # input_size = feature_size per timestamp outputted by the CNNs
        # hidden_size = gru_hidden_dim
        self._biGRU = nn.GRU(input_size=12, hidden_size=gru_units, num_layers=1, bidirectional=True, batch_first=True)

        self._biGru_activation_do = nn.Sequential(
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )

        match attention_type:
            case "v1":
                # Own MHA implementation with random query initialization
                head_dims = use_reduced_head_dims if use_reduced_head_dims is not None else False
                attention_activation = attention_activation_function if attention_activation_function is not None else "softmax"
                special_init = attention_special_init if attention_special_init is not None else False
                self._multi_head_contextual_attention = MultiHeadContextualAttention(d_model=2 * gru_units,
                                                                                     dropout=dropout_attention,
                                                                                     heads=heads,
                                                                                     discard_FC_before_MH=discard_FC_before_MH,
                                                                                     use_reduced_head_dims=head_dims,
                                                                                     attention_activation_function=attention_activation,
                                                                                     special_init=special_init)
            case "v2":
                # Torch MHA implementation with random query initialization
                self._multi_head_contextual_attention = MultiHeadContextualAttentionV2(d_model=2 * gru_units,
                                                                                       dropout=dropout_attention,
                                                                                       heads=heads,
                                                                                       discard_FC_before_MH=discard_FC_before_MH)
            case "v3":
                # Torch MHA implementation +  buggy query initialization based on the mean of the BiGRU output
                self._multi_head_contextual_attention = MultiHeadContextualAttentionV3(d_model=2 * gru_units,
                                                                                        dropout=dropout_attention,
                                                                                        heads=heads,
                                                                                        discard_FC_before_MH=discard_FC_before_MH)
            case "v4":
                # Torch MHA implementation +  Learnable query projection with Linear layer
                self._multi_head_contextual_attention = MultiHeadContextualAttentionV4(d_model=2 * gru_units,
                                                                                       dropout=dropout_attention,
                                                                                       heads=heads,
                                                                                       discard_FC_before_MH=discard_FC_before_MH)
            case "v5":
                # Torch MHA implementation +  Method "initialize_weights()" for query initialization
                self._multi_head_contextual_attention = MultiHeadContextualAttentionV5(d_model=2 * gru_units,
                                                                                       dropout=dropout_attention,
                                                                                       heads=heads,
                                                                                       discard_FC_before_MH=discard_FC_before_MH)
            case _:
                raise ValueError(f"Attention type {attention_type} not supported!")

        self._batchNorm = nn.Sequential(
            # The batch normalization layer has 24*2=48 trainable and 24*2=48 non-trainable parameters
            nn.BatchNorm1d(gru_units * 2),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )

        self._fcn = nn.Linear(in_features=gru_units * 2, out_features=num_classes)

        if apply_final_activation:
            self._final_activation = nn.Sigmoid() if multi_label_training else nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Plot the first sample from the batch
        # plot_record_from_np_array(x[1].detach().numpy())
        x = self._conv_block1(x)
        # plot_record_from_np_array(x[1].detach().numpy())
        x = self._conv_block2(x)
        # plot_record_from_np_array(x[1].detach().numpy())
        x = self._conv_block3(x)
        # plot_record_from_np_array(x[1].detach().numpy())
        x = self._conv_block4(x)
        # plot_record_from_np_array(x[1].detach().numpy())
        x = self._conv_block5(x)
        # plot_record_from_np_array(x[1].detach().numpy())
        # Do not use view() or reshape() to swap dimensions of tensors!
        # view() and reshape() nevertheless have their purpose, for example, to flatten tensors.
        # See:https://discuss.pytorch.org/t/for-beginners-do-not-use-view-or-reshape-to-swap-dimensions-of-tensors/75524
        x = x.permute(0, 2, 1)  # switch seq_length and feature_size for the BiGRU
        x, last_hidden_state = self._biGRU(x)
        x = self._biGru_activation_do(x)
        x = self._multi_head_contextual_attention(x)
        x = self._batchNorm(x)
        x = self._fcn(x)
        if self._apply_final_activation:
            return self._final_activation(x)
        else:
            return x


if __name__ == "__main__":
    model = BaselineModelWithMHAttentionV2(apply_final_activation=True, multi_label_training=True, gru_units=12)
    summary(model, input_size=(2, 12, 15000), col_names=["input_size", "output_size", "num_params"])
