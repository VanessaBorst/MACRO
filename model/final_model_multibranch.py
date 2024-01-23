from abc import ABC

import torch.nn as nn
import torch
from torchinfo import summary

from base import BaseModel
from layers.BasicBlock1dWithNorm import BasicBlock1dWithNorm
from layers.BasicBlock1dWithNormPreActivationDesign import BasicBlock1dWithNormPreactivation
from layers.ContextualAttention import MultiHeadContextualAttention


class FinalModelMultiBranch(BaseModel):

    def __init__(self,
                 multi_branch_heads=2,
                 first_conv_reduction_kernel_size=3,
                 second_conv_reduction_kernel_size=3,
                 third_conv_reduction_kernel_size=3,
                 vary_channels_lighter_version=True,
                 discard_FC_before_MH=True,
                 branchNet_gru_units=24,
                 branchNet_heads=2,
                 multi_label_training=True,
                 num_classes=9):
        """

        @return:
        """
        super().__init__()

        final_model_single_leads = [FinalModelBranchNet(
            gru_units=branchNet_gru_units,
            heads=branchNet_heads,
            vary_channels_lighter_version=vary_channels_lighter_version,
            discard_FC_before_MH=discard_FC_before_MH,
            multi_label_training=multi_label_training,
            input_channel=1) for _ in range(12)]

        self._final_model_single_leads = nn.Sequential(*final_model_single_leads)

        # branchNet_gru_units -> in_channels -> channel_reduction_per_conv -> channel_seq
        # 12 -> in_channels = 288 -> 92 -> 288, 200, 112, 24
        # 24 -> in_channels = 576 -> 188 -> 576, 392, 208, 24
        # 32 -> in_channels = 768 -> 252 -> 768, 520, 272, 24
        in_channels = branchNet_gru_units * 2 * 12
        # The out_channels should be reduced to 12 in three steps
        first_out_channels = in_channels - (in_channels - 24) // 3
        second_out_channels = in_channels - 2 * (in_channels - 24) // 3
        third_out_channels = 24

        # # Define the function to calculate same padding (with stride 1 and dilation 1 !!!)
        # def calculate_same_padding(kernel_size):
        #     if kernel_size % 2 == 0:
        #         # If stride is 1 and the kernel_size even, the additional one-sided 0 is used to keep/half dimension
        #         return kernel_size // 2
        #     else:
        #         # If the kernel size is odd, the padding is the same on both sides, no one-sided padding is needed
        #         return (kernel_size - 1) // 2
        #
        # # Apply padding to ensure that the sequence length stays the same
        # # If stride is 1 and the kernel_size is even, an additional one-sided 0 is needed to keep dimensions
        # self._one_sided_padding = nn.ConstantPad1d((0, 1), 0)
        #
        # self._half_padding_first_kernel = nn.ConstantPad1d((calculate_same_padding(first_conv_reduction_kernel_size), calculate_same_padding(first_conv_reduction_kernel_size)), 0)
        # self._half_padding_second_kernel = nn.ConstantPad1d((calculate_same_padding(second_conv_reduction_kernel_size), calculate_same_padding(second_conv_reduction_kernel_size)), 0)
        # self._half_padding_third_kernel = nn.ConstantPad1d((calculate_same_padding(third_conv_reduction_kernel_size), calculate_same_padding(third_conv_reduction_kernel_size)),0)

        self._convolutional_reduction_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=first_out_channels,
                      kernel_size=first_conv_reduction_kernel_size, padding="same"),
            nn.BatchNorm1d(first_out_channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=first_out_channels, out_channels=second_out_channels,
                      kernel_size=second_conv_reduction_kernel_size, padding="same"),
            nn.BatchNorm1d(second_out_channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=second_out_channels, out_channels=third_out_channels,
                      kernel_size=third_conv_reduction_kernel_size, padding="same"),
            nn.LeakyReLU(0.3)
        )

        # d_model is 24 in this case, before it was 2 * gru_units
        self._multi_head_contextual_attention = MultiHeadContextualAttention(d_model=third_out_channels,
                                                                             dropout=0.3,
                                                                             heads=multi_branch_heads,
                                                                             discard_FC_before_MH=discard_FC_before_MH)

        self._batchNorm = nn.Sequential(
            nn.BatchNorm1d(third_out_channels),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )

        # Input: d_model of the MH mechanism
        self._fcn = nn.Linear(in_features=third_out_channels, out_features=num_classes)
        # apply final activation is false by default

    def forward(self, x):
        """

        @param x:  # x has shape [batch_size, 12, seq_len]
        @return:
        """
        # Format will be a list of length 12 containing of tuples of the following format
        # (final_output, biGRU_output) with
        # final_output: batch_size x 9
        # biGRU_output: batch_size x (seq_len/(2^5)) x (branchNet_gru_units * 2)
        single_lead_results = [
            self._final_model_single_leads[i](x[:, i, None, :]) for i in range(12)
        ]
        # Concatenate the results of the single lead branches in channel dimension
        # Cat (BiGRU_1, ..., BiGRU_12)
        # BiGRU output has shape batch_size x (seq_len/(2^5)) x (branchNet_gru_units * 2)
        # Hence, after concat, the shape is: batch_size x (seq_len/(2^5)) x (branchNet_gru_units * 2 * 12)
        x = torch.cat([single_lead_results[i][1] for i in range(12)], dim=2)

        # x has shape [batch_size, seq_len, 12 * 2 * branchNet_gru_units]
        # switch seq_length and feature_size after the BiGRUs again for further convolutional processing
        x = x.permute(0, 2, 1)

        # x -> batch_size  x (branchNet_gru_units * 2 * 12) x 2250
        # (branchNet_gru_units * 2 * 12) is 288, 576 or 768
        # Conv1 -> Reduction to 196, 388, or 516 channels
        # Conv2 -> Reduction to 104, 200, or 264 channels
        # Conv3 -> Reduction to 12 channels
        x = self._convolutional_reduction_block(x)

        # x has shape [bs, 12, len*], len* = 2250
        # switch seq_length and feature_size again for getting a weighted sum over the features of all time steps
        x = x.permute(0, 2, 1)
        # x has shape [bs, len*, 12]
        x = self._multi_head_contextual_attention(x)

        # x has shape [bs, 12]
        x = self._batchNorm(x)
        x = self._fcn(x)
        # x has shape [bs, num_classes]

        return x, [single_lead_results[i][0] for i in range(12)]


class FinalModelBranchNet(BaseModel):
    def __init__(self,
                 apply_final_activation=False,
                 multi_label_training=True,
                 input_channel=12,
                 num_classes=9,
                 drop_out_first_conv_blocks=0.2,
                 drop_out_second_conv_blocks=0.2,
                 drop_out_gru=0.2,
                 dropout_last_bn=0.2,
                 mid_kernel_size_first_conv_blocks=3,
                 mid_kernel_size_second_conv_blocks=3,
                 last_kernel_size_first_conv_blocks=24,
                 last_kernel_size_second_conv_blocks=48,
                 stride_first_conv_blocks=2,
                 stride_second_conv_blocks=2,
                 down_sample="conv",
                 vary_channels=True,
                 vary_channels_lighter_version=False,
                 pos_skip="all",  # MA model: "not_last",
                 norm_type="BN",
                 norm_pos="all",
                 norm_before_act=True,
                 use_pre_activation_design=True,
                 use_pre_conv=True,
                 pre_conv_kernel=16,
                 gru_units=12,
                 dropout_attention=0.3,  # MA model: 0.2,
                 heads=3,
                 discard_FC_before_MH=False):
        """
        :param apply_final_activation: whether the Sigmoid(sl) or the LogSoftmax(ml) should be applied at the end
        :param multi_label_training: if true, Sigmoid is used as final activation, else the LogSoftmax
        :param num_classes: Num of classes to classify
        :param num_cnn_blocks: Num of CNN blocks to use
        """
        super().__init__()
        self._apply_final_activation = apply_final_activation

        assert down_sample == "conv" or down_sample == "max_pool" or down_sample == "avg_pool", \
            "Downsampling should either be conv or max_pool or avg_pool"

        assert not vary_channels_lighter_version or vary_channels_lighter_version == vary_channels == True, \
            "vary_channels_lighter_version only has an effect if vary_channels is True as well"

        if use_pre_activation_design:
            assert pos_skip == "all" or pos_skip == "not_last", "For the preactivation design, ''not first'' is no valid" \
                                                                "option for the skip connections! Choose between ''all'' " \
                                                                "and ''not last''"
        else:
            # assert use_pre_conv is None or use_pre_conv is False, "Pre-Conv not possible without pre-activation design"
            assert norm_before_act is None or norm_before_act is True, "Norm after activation not possible without " \
                                                                       "preactivation design"

        self._pos_skip = pos_skip
        self._use_pre_conv = use_pre_conv
        self._use_pre_activation_design = use_pre_activation_design

        if vary_channels:
            # Old version -> when used in multi-branch setting, reduce the amount of channels
            # out_channel_block_1 = 24
            # out_channel_block_2 = 48
            # out_channel_block_3 = 48
            # out_channel_block_4 = 24
            # out_channel_block_5 = 12
            if vary_channels_lighter_version:
                out_channel_block_1 = 12
                out_channel_block_2 = 24
                out_channel_block_3 = 24
                out_channel_block_4 = 12
                out_channel_block_5 = 1
            else:
                out_channel_block_1 = 12
                out_channel_block_2 = 24
                out_channel_block_3 = 48
                out_channel_block_4 = 24
                out_channel_block_5 = 12
        else:
            out_channel_block_1 = out_channel_block_2 = out_channel_block_3 \
                = out_channel_block_4 = out_channel_block_5 = 12

        if use_pre_activation_design and use_pre_conv:
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

        if use_pre_activation_design:
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

        else:
            if pos_skip == "all" or pos_skip == "not_last":
                self._first_conv_block_1 = BasicBlock1dWithNorm(in_channels=input_channel,
                                                                out_channels=out_channel_block_1,
                                                                mid_kernels_size=mid_kernel_size_first_conv_blocks,
                                                                last_kernel_size=last_kernel_size_first_conv_blocks,
                                                                stride=stride_first_conv_blocks,
                                                                down_sample=down_sample,
                                                                drop_out=drop_out_first_conv_blocks,
                                                                skips_active=True,
                                                                norm_type=norm_type, norm_pos=norm_pos)
            elif pos_skip == "not_first":
                self._first_conv_block_1 = BasicBlock1dWithNorm(in_channels=input_channel,
                                                                out_channels=out_channel_block_1,
                                                                mid_kernels_size=mid_kernel_size_first_conv_blocks,
                                                                last_kernel_size=last_kernel_size_first_conv_blocks,
                                                                stride=stride_first_conv_blocks,
                                                                down_sample=down_sample,
                                                                drop_out=drop_out_first_conv_blocks,
                                                                skips_active=False,
                                                                norm_type=norm_type, norm_pos=norm_pos)

            self._first_conv_block_2 = BasicBlock1dWithNorm(in_channels=out_channel_block_1,
                                                            out_channels=out_channel_block_2,
                                                            mid_kernels_size=mid_kernel_size_first_conv_blocks,
                                                            last_kernel_size=last_kernel_size_first_conv_blocks,
                                                            stride=stride_first_conv_blocks,
                                                            down_sample=down_sample,
                                                            drop_out=drop_out_first_conv_blocks,
                                                            skips_active=True,
                                                            norm_type=norm_type, norm_pos=norm_pos)
            self._first_conv_block_3 = BasicBlock1dWithNorm(in_channels=out_channel_block_2,
                                                            out_channels=out_channel_block_3,
                                                            mid_kernels_size=mid_kernel_size_first_conv_blocks,
                                                            last_kernel_size=last_kernel_size_first_conv_blocks,
                                                            stride=stride_first_conv_blocks,
                                                            down_sample=down_sample,
                                                            drop_out=drop_out_first_conv_blocks,
                                                            skips_active=True,
                                                            norm_type=norm_type, norm_pos=norm_pos)
            self._first_conv_block_4 = BasicBlock1dWithNorm(in_channels=out_channel_block_3,
                                                            out_channels=out_channel_block_4,
                                                            mid_kernels_size=mid_kernel_size_first_conv_blocks,
                                                            last_kernel_size=last_kernel_size_first_conv_blocks,
                                                            stride=stride_first_conv_blocks,
                                                            down_sample=down_sample,
                                                            drop_out=drop_out_first_conv_blocks,
                                                            skips_active=True,
                                                            norm_type=norm_type, norm_pos=norm_pos)

            # Second Type of Conv Blocks
            if pos_skip == "all" or pos_skip == "not_first":
                self._second_conv_block_1 = BasicBlock1dWithNorm(in_channels=out_channel_block_4,
                                                                 out_channels=out_channel_block_5,
                                                                 mid_kernels_size=mid_kernel_size_second_conv_blocks,
                                                                 last_kernel_size=last_kernel_size_second_conv_blocks,
                                                                 stride=stride_second_conv_blocks,
                                                                 down_sample=down_sample,
                                                                 drop_out=drop_out_second_conv_blocks,
                                                                 skips_active=True,
                                                                 norm_type=norm_type, norm_pos=norm_pos)
            elif pos_skip == "not_last":
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
        self._biGRU = nn.GRU(input_size=out_channel_block_5, hidden_size=gru_units,
                             num_layers=1, bidirectional=True, batch_first=True)

        self._biGru_activation_do = nn.Sequential(
            nn.LeakyReLU(0.3),
            nn.Dropout(drop_out_gru)
        )

        self._multi_head_contextual_attention = MultiHeadContextualAttention(d_model=2 * gru_units,
                                                                             dropout=dropout_attention,
                                                                             heads=heads,
                                                                             discard_FC_before_MH=discard_FC_before_MH)

        self._batchNorm = nn.Sequential(
            # The batch normalization layer has 24*2=48 trainable and 24*2=48 non-trainable parameters
            nn.BatchNorm1d(gru_units * 2),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout_last_bn)
        )

        self._fcn = nn.Linear(in_features=gru_units * 2, out_features=num_classes)

        if apply_final_activation:
            self._final_activation = nn.Sigmoid() if multi_label_training else nn.LogSoftmax(dim=1)

    def forward(self, x):
        if self._use_pre_activation_design:
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
        else:
            x = self._first_conv_block_1(x)
            x = self._first_conv_block_2(x)
            x = self._first_conv_block_3(x)
            x = self._first_conv_block_4(x)
            x = self._second_conv_block_1(x)

        x = x.permute(0, 2, 1)  # switch seq_length and feature_size for the BiGRU
        x, last_hidden_state = self._biGRU(x)
        x = self._biGru_activation_do(x)
        # Copy BiGRU output for passing it to the MultiBranchNet concatenation
        single_branch_biGRU_output = x
        x = self._multi_head_contextual_attention(x)
        x = self._batchNorm(x)
        x = self._fcn(x)
        if self._apply_final_activation:
            return self._final_activation(x), single_branch_biGRU_output
        else:
            return x, single_branch_biGRU_output


if __name__ == "__main__":
    model = FinalModelMultiBranch(multi_branch_heads=2)
    summary(model, input_size=(2, 12, 72000), col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=5)

    model_part = FinalModelBranchNet(apply_final_activation=False,
                                     multi_label_training=True,
                                     input_channel=1,
                                     num_classes=9,
                                     drop_out_first_conv_blocks=0.2,
                                     drop_out_second_conv_blocks=0.2,
                                     drop_out_gru=0.2,
                                     dropout_last_bn=0.2,
                                     mid_kernel_size_first_conv_blocks=3,
                                     mid_kernel_size_second_conv_blocks=3,
                                     last_kernel_size_first_conv_blocks=24,
                                     last_kernel_size_second_conv_blocks=48,
                                     stride_first_conv_blocks=2,
                                     stride_second_conv_blocks=2,
                                     down_sample="conv",
                                     vary_channels=True,
                                     pos_skip="all",  # MA model: "not_last",
                                     norm_type="BN",
                                     norm_pos="all",
                                     norm_before_act=True,
                                     use_pre_activation_design=True,
                                     use_pre_conv=True,
                                     pre_conv_kernel=16,
                                     gru_units=24,
                                     dropout_attention=0.3,  # MA model: 0.2,
                                     heads=2,
                                     discard_FC_before_MH=True)
    summary(model_part, input_size=(2, 1, 72000), col_names=["input_size", "output_size", "kernel_size", "num_params"])
