import torch.nn as nn
from torchinfo import summary

from base import BaseModel
from layers.ContextualAttention import ContextualAttention
from layers.BasicBlock1d import BasicBlock1d


class BaselineModelWithSkipConnections(BaseModel):
    def __init__(self, apply_final_activation, multi_label_training, down_sample="conv",
                 num_blocks=4, input_channel=12, num_classes=9):
        """
        :param apply_final_activation: whether the Sigmoid(sl) or the LogSoftmax(ml) should be applied at the end
        :param multi_label_training: if true, Sigmoid is used as final activation, else the LogSoftmax
        :param num_classes: Num of classes to classify
        :param num_blocks: Num of CNN blocks to use
        """
        super().__init__()
        self._apply_final_activation = apply_final_activation

        assert down_sample == "conv" or down_sample == "pool", "Downsampling should either be conv or pool"

        self.inplanes = input_channel
        self._conv_blocks = nn.ModuleList([
            nn.Sequential(
                BasicBlock1d(in_channels=self.inplanes, out_channels=12,
                             last_kernel_size=24, down_sample=down_sample)
            ) for _ in range(num_blocks)]
        )
        # Last block has a convolutional pooling with kernel size 48!
        self._last_conv_block = BasicBlock1d(in_channels=12, out_channels=12,
                                             last_kernel_size=48, down_sample=down_sample)

        # Without last option the input would have to be (seq_len, batch, input_size)
        # With batch_first it can be of the shape (batch, seq_len, input/feature_size)
        # input_size = feature_size per timestamp outputted by the CNNs
        # hidden_size = gru_hidden_dim
        self._biGRU = nn.GRU(input_size=12, hidden_size=12, num_layers=1, bidirectional=True, batch_first=True)

        self._biGru_activation_do = nn.Sequential(
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )

        self._contextual_attention = ContextualAttention(gru_dimension=12, attention_dimension=24)

        self._batchNorm = nn.Sequential(
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )

        self._fcn = nn.Linear(in_features=24, out_features=num_classes)

        if apply_final_activation:
            self._final_activation = nn.Sigmoid() if multi_label_training else nn.LogSoftmax(dim=1)

    def forward(self, x):
        for _, conv_block in enumerate(self._conv_blocks):
            x = conv_block(x)
        x = self._last_conv_block(x)
        x = x.permute(0, 2, 1)  # switch seq_length and feature_size for the BiGRU
        x, last_hidden_state = self._biGRU(x)
        x = self._biGru_activation_do(x)
        x, attention_weights = self._contextual_attention(x)
        x = self._batchNorm(x)
        x = self._fcn(x)
        if self._apply_final_activation:
            return self._final_activation(x), attention_weights
        else:
            return x, attention_weights


if __name__ == "__main__":
    model = BaselineModelWithSkipConnections(down_sample="conv", apply_final_activation=True, multi_label_training=True)
    summary(model, input_size=(2, 12, 72000), col_names=["input_size", "output_size", "num_params"])
