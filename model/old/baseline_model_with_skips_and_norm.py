import torch.nn as nn
from torchinfo import summary

from base import BaseModel
from layers.BasicBlock1dWithNorm import BasicBlock1dWithNorm
from layers.ContextualAttention import ContextualAttention


# /1_5_0.2_0.3_24_64_3_3_13_44_1_1_conv_23-22-57
class BaselineModelWithSkipConnectionsAndNorm(BaseModel):
    def __init__(self, apply_final_activation, multi_label_training, input_channel=12, num_classes=9,
                 num_first_conv_blocks=4, num_second_conv_blocks=1,
                 drop_out_first_conv_blocks=0.2, drop_out_second_conv_blocks=0.2,
                 out_channel_first_conv_blocks=12, out_channel_second_conv_blocks=12,
                 mid_kernel_size_first_conv_blocks=3, mid_kernel_size_second_conv_blocks=3,
                 last_kernel_size_first_conv_blocks=24, last_kernel_size_second_conv_blocks=48,
                 stride_first_conv_blocks=2, stride_second_conv_blocks=2,
                 down_sample="conv", norm_type="BN", norm_pos="last", norm_before_act=True):
        """
        :param apply_final_activation: whether the Sigmoid(sl) or the LogSoftmax(ml) should be applied at the end
        :param multi_label_training: if true, Sigmoid is used as final activation, else the LogSoftmax
        :param num_classes: Num of classes to classify
        :param num_blocks: Num of CNN blocks to use
        """
        super().__init__()
        self._apply_final_activation = apply_final_activation

        assert down_sample == "conv" or down_sample == "max_pool" or down_sample == "avg_pool", \
            "Downsampling should either be conv or max_pool or avg_pool"

        self._first_conv_blocks_1 = BasicBlock1dWithNorm(in_channels=input_channel,
                                                         out_channels=out_channel_first_conv_blocks,
                                                         mid_kernels_size=mid_kernel_size_first_conv_blocks,
                                                         last_kernel_size=last_kernel_size_first_conv_blocks,
                                                         stride=stride_first_conv_blocks,
                                                         down_sample=down_sample,
                                                         drop_out=drop_out_first_conv_blocks,
                                                         norm_type=norm_type, norm_pos=norm_pos,
                                                         norm_before_act=norm_before_act)
        self._first_conv_blocks_2 = nn.ModuleList([
            BasicBlock1dWithNorm(in_channels=out_channel_first_conv_blocks,
                                 out_channels=out_channel_first_conv_blocks,
                                 mid_kernels_size=mid_kernel_size_first_conv_blocks,
                                 last_kernel_size=last_kernel_size_first_conv_blocks,
                                 stride=stride_first_conv_blocks,
                                 down_sample=down_sample,
                                 drop_out=drop_out_first_conv_blocks,
                                 norm_type=norm_type, norm_pos=norm_pos,
                                 norm_before_act=norm_before_act)
            for _ in range(num_first_conv_blocks - 1)]
        )

        self._second_conv_blocks_1 = BasicBlock1dWithNorm(in_channels=out_channel_first_conv_blocks,
                                                          out_channels=out_channel_second_conv_blocks,
                                                          mid_kernels_size=mid_kernel_size_second_conv_blocks,
                                                          last_kernel_size=last_kernel_size_second_conv_blocks,
                                                          stride=stride_second_conv_blocks,
                                                          down_sample=down_sample,
                                                          drop_out=drop_out_second_conv_blocks,
                                                          norm_type=norm_type, norm_pos=norm_pos,
                                                          norm_before_act=norm_before_act)
        self._second_conv_blocks_2 = nn.ModuleList([
            BasicBlock1dWithNorm(in_channels=out_channel_second_conv_blocks,
                                 out_channels=out_channel_second_conv_blocks,
                                 mid_kernels_size=mid_kernel_size_second_conv_blocks,
                                 last_kernel_size=last_kernel_size_second_conv_blocks,
                                 stride=stride_second_conv_blocks,
                                 down_sample=down_sample,
                                 drop_out=drop_out_second_conv_blocks,
                                 norm_type=norm_type, norm_pos=norm_pos,
                                 norm_before_act=norm_before_act)
            for _ in range(num_second_conv_blocks - 1)]
        )

        # Without last option the input would have to be (seq_len, batch, input_size)
        # With batch_first it can be of the shape (batch, seq_len, input/feature_size)
        # input_size = feature_size per timestamp outputted by the CNNs
        # hidden_size = gru_hidden_dim
        self._biGRU = nn.GRU(input_size=out_channel_second_conv_blocks, hidden_size=12,
                             num_layers=1, bidirectional=True, batch_first=True)

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
        x = self._first_conv_blocks_1(x)
        for _, conv_block in enumerate(self._first_conv_blocks_2):
            x = conv_block(x)
        x = self._second_conv_blocks_1(x)
        for _, conv_block in enumerate(self._second_conv_blocks_2):
            x = conv_block(x)

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
    model = BaselineModelWithSkipConnectionsAndNorm(apply_final_activation=False,
                                                    multi_label_training=True, input_channel=12,
                                                    down_sample="conv",
                                                    drop_out_first_conv_blocks=0.2,
                                                    drop_out_second_conv_blocks=0.3,
                                                    last_kernel_size_first_conv_blocks=13,
                                                    last_kernel_size_second_conv_blocks=44,
                                                    mid_kernel_size_first_conv_blocks=3,
                                                    mid_kernel_size_second_conv_blocks=3,
                                                    num_first_conv_blocks=2,
                                                    num_second_conv_blocks=6,
                                                    out_channel_first_conv_blocks=24,
                                                    out_channel_second_conv_blocks=64,
                                                    stride_first_conv_blocks=2,
                                                    stride_second_conv_blocks=2
                                                    )
    # 2_6_0.2_0.3_24_64_3_3_13_44_2_2_conv_23

    summary(model, input_size=(2, 12, 72000), col_names=["input_size", "output_size", "num_params"])
