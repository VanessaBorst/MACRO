import torch.nn as nn
from torchinfo import summary

from base import BaseModel
from layers.BasicBlock1dWithNorm import BasicBlock1dWithNorm
from layers.BasicBlock1dWithNormPreActivationDesign import BasicBlock1dWithNormPreactivation
from layers.ContextualAttention import ContextualAttention


class BaselineModelWithSkipConnectionsAndNormV2PreActivation(BaseModel):
    def __init__(self, apply_final_activation, multi_label_training, input_channel=12, num_classes=9,
                 drop_out_first_conv_blocks=0.2, drop_out_second_conv_blocks=0.2,
                 mid_kernel_size_first_conv_blocks=3, mid_kernel_size_second_conv_blocks=3,
                 last_kernel_size_first_conv_blocks=24, last_kernel_size_second_conv_blocks=48,
                 stride_first_conv_blocks=2, stride_second_conv_blocks=2,
                 down_sample="conv", vary_channels=True, pos_skip="not_last",
                 norm_type="BN", norm_pos="all", norm_before_act=True, use_pre_conv=True,
                 pre_conv_kernel=16):
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

        assert pos_skip == "all" or pos_skip == "not_last", "For the preactivation design, ''not first'' is no valid" \
                                                            "option for the skip connections! Choose between ''all'' " \
                                                            "and ''not last''"
        assert norm_type == "BN" or norm_type == "IN", "For the preactivation design, only BN and IN are supported"


        self._pos_skip = pos_skip
        self._use_pre_conv = use_pre_conv

        if vary_channels:
            out_channel_block_1 = 24
            out_channel_block_2 = 48
            out_channel_block_3 = 48
            out_channel_block_4 = 24
            out_channel_block_5 = 12
        else:
            out_channel_block_1 = out_channel_block_2 = out_channel_block_3 \
                = out_channel_block_4 = out_channel_block_5 = 12

        if use_pre_conv:
            # Start with a convolution with stride 1, which keeps the channel amount constant
            if norm_type == "BN":
                starting_norm = nn.BatchNorm1d(num_features=input_channel)
            elif norm_type == "IN":
                starting_norm = nn.InstanceNorm1d(num_features=input_channel, affine=True)

            if norm_before_act:
                self._starting_conv = nn.Sequential(
                    nn.Conv1d(in_channels=input_channel, out_channels=input_channel, kernel_size=pre_conv_kernel),
                    starting_norm,
                    nn.LeakyReLU(0.3)
                )
            else:
                self._starting_conv = nn.Sequential(
                    nn.Conv1d(in_channels=input_channel, out_channels=input_channel, kernel_size=pre_conv_kernel),
                    nn.LeakyReLU(0.3),
                    starting_norm
                )

        self._first_conv_block_1 = BasicBlock1dWithNormPreactivation(in_channels=input_channel,
                                                                     out_channels=out_channel_block_1,
                                                                     mid_kernels_size=mid_kernel_size_first_conv_blocks,
                                                                     last_kernel_size=last_kernel_size_first_conv_blocks,
                                                                     stride=stride_first_conv_blocks,
                                                                     down_sample=down_sample,
                                                                     drop_out=drop_out_first_conv_blocks,
                                                                     norm_type=norm_type, norm_pos=norm_pos,
                                                                     norm_before_act=norm_before_act)

        self._first_conv_block_2 = BasicBlock1dWithNormPreactivation(in_channels=out_channel_block_1,
                                                                     out_channels=out_channel_block_2,
                                                                     mid_kernels_size=mid_kernel_size_first_conv_blocks,
                                                                     last_kernel_size=last_kernel_size_first_conv_blocks,
                                                                     stride=stride_first_conv_blocks,
                                                                     down_sample=down_sample,
                                                                     drop_out=drop_out_first_conv_blocks,
                                                                     norm_type=norm_type, norm_pos=norm_pos,
                                                                     norm_before_act=norm_before_act)
        self._first_conv_block_3 = BasicBlock1dWithNormPreactivation(in_channels=out_channel_block_2,
                                                                     out_channels=out_channel_block_3,
                                                                     mid_kernels_size=mid_kernel_size_first_conv_blocks,
                                                                     last_kernel_size=last_kernel_size_first_conv_blocks,
                                                                     stride=stride_first_conv_blocks,
                                                                     down_sample=down_sample,
                                                                     drop_out=drop_out_first_conv_blocks,
                                                                     norm_type=norm_type, norm_pos=norm_pos,
                                                                     norm_before_act=norm_before_act)
        self._first_conv_block_4 = BasicBlock1dWithNormPreactivation(in_channels=out_channel_block_3,
                                                                     out_channels=out_channel_block_4,
                                                                     mid_kernels_size=mid_kernel_size_first_conv_blocks,
                                                                     last_kernel_size=last_kernel_size_first_conv_blocks,
                                                                     stride=stride_first_conv_blocks,
                                                                     down_sample=down_sample,
                                                                     drop_out=drop_out_first_conv_blocks,
                                                                     norm_type=norm_type, norm_pos=norm_pos,
                                                                     norm_before_act=norm_before_act)

        # Second Type of Conv Blocks
        if pos_skip == "all":
            self._second_conv_block_1 = BasicBlock1dWithNormPreactivation(in_channels=out_channel_block_4,
                                                                          out_channels=out_channel_block_5,
                                                                          mid_kernels_size=mid_kernel_size_second_conv_blocks,
                                                                          last_kernel_size=last_kernel_size_second_conv_blocks,
                                                                          stride=stride_second_conv_blocks,
                                                                          down_sample=down_sample,
                                                                          drop_out=drop_out_second_conv_blocks,
                                                                          norm_type=norm_type, norm_pos=norm_pos,
                                                                          norm_before_act=norm_before_act)
        elif pos_skip == "not_last":
            # Use the normal block without pre-activation and deactivate the skip connections
            # Attention: This must be handled in the forward, since the normal block exspects a single
            # input value, but the pre-activation block output a tuple (out, residuals)
            self._second_conv_block_1 = BasicBlock1dWithNorm(in_channels=out_channel_block_4,
                                                             out_channels=out_channel_block_5,
                                                             mid_kernels_size=mid_kernel_size_second_conv_blocks,
                                                             last_kernel_size=last_kernel_size_second_conv_blocks,
                                                             stride=stride_second_conv_blocks,
                                                             down_sample=down_sample,
                                                             drop_out=drop_out_second_conv_blocks,
                                                             skips_active=False,
                                                             norm_type=norm_type, norm_pos=norm_pos)

        # Without last option the input would have to be (seq_len, batch, input_size)
        # With batch_first it can be of the shape (batch, seq_len, input/feature_size)
        # input_size = feature_size per timestamp outputted by the CNNs
        # hidden_size = gru_hidden_dim
        self._biGRU = nn.GRU(input_size=out_channel_block_5, hidden_size=12,
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
        if self._use_pre_conv:
            x = self._starting_conv(x)
        # Start with the residual blocks
        x = self._first_conv_block_1((x, x))
        x = self._first_conv_block_2(x)
        x = self._first_conv_block_3(x)
        x = self._first_conv_block_4(x)
        # The last block needs to be handled separately, depending if skips should be used as well
        if self._pos_skip == "all":
            x = self._second_conv_block_1(x)
        elif self._pos_skip == "not_last":
            x, residuals = x
            x = self._second_conv_block_1(x)

        # If the last block uses skips, it return a tuple
        if self._pos_skip == "all":
            x, residuals = x

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
    model = BaselineModelWithSkipConnectionsAndNormV2PreActivation(apply_final_activation=False,
                                                                   multi_label_training=True, input_channel=12,
                                                                   num_classes=9,
                                                                   drop_out_first_conv_blocks=0.2,
                                                                   drop_out_second_conv_blocks=0.2,
                                                                   last_kernel_size_first_conv_blocks=24,
                                                                   last_kernel_size_second_conv_blocks=48,
                                                                   mid_kernel_size_first_conv_blocks=3,
                                                                   mid_kernel_size_second_conv_blocks=3,
                                                                   # num_first_conv_blocks=2,
                                                                   # num_second_conv_blocks=6,
                                                                   # out_channel_first_conv_blocks=24,
                                                                   # out_channel_second_conv_blocks=64,
                                                                   stride_first_conv_blocks=2,
                                                                   stride_second_conv_blocks=2,
                                                                   down_sample="conv",
                                                                   vary_channels=True,
                                                                   pos_skip="not_last",
                                                                   norm_type="BN", norm_pos="all", norm_before_act=True
                                                                   )
    # 2_6_0.2_0.3_24_64_3_3_13_44_2_2_conv_23

    summary(model, input_size=(2, 12, 72000), col_names=["input_size", "output_size", "num_params"])


