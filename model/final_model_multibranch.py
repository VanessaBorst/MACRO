from abc import ABC
from typing import Any

import torch.nn as nn
import torch
from torchinfo import summary

from base import BaseModel
from layers.ContextualAttention import MultiHeadContextualAttention
from model.final_model import FinalModel


class FinalModelMultiBranch(BaseModel):

    def __init__(self,
                 # BranchNet specifics
                 branchNet_reduce_channels=True,
                 branchNet_gru_units=12,
                 branchNet_heads=6,
                 branchNet_attention_dropout=0.2,
                 # Multibranch specifics
                 multi_branch_heads=2,
                 multi_branch_attention_dropout=0.2,
                 use_conv_reduction_block=False,
                 conv_reduction_first_kernel_size=None,  # before: 3
                 conv_reduction_second_kernel_size=None,  # before: 3
                 conv_reduction_third_kernel_size=None,  # before: 3
                 # General parameters
                 apply_final_activation=False,  # Set once for both, BranchNet and MultiBranchNet
                 multi_label_training=True,
                 num_classes=9):
        """

        @return:
        """
        super().__init__()

        assert use_conv_reduction_block or (conv_reduction_first_kernel_size is None and
                                            conv_reduction_second_kernel_size is None and
                                            conv_reduction_third_kernel_size is None), \
            "If use_conv_reduction_block is False, the kernel sizes should be None"

        self._use_conv_reduction_block = use_conv_reduction_block
        self._apply_final_activation = apply_final_activation

        final_model_single_leads = [
            FinalModel(
                # General parameters
                apply_final_activation=apply_final_activation,
                multi_label_training=multi_label_training,
                input_channel=1,
                num_classes=num_classes,
                # CNN-related parameters (almost all as in the default config)
                pos_skip="all",
                # GRU and MHA-related parameters
                gru_units=branchNet_gru_units,
                heads=branchNet_heads,
                dropout_attention=branchNet_attention_dropout,
                # The following parameters are fixed for the paper
                use_reduced_head_dims=True,
                attention_activation_function="entmax15",
                # Multibranch-specific parameters
                act_as_branch_net=True,
                vary_channels_lighter_version=branchNet_reduce_channels
            ) for _ in range(12)]

        self._final_model_single_leads = nn.Sequential(*final_model_single_leads)

        if self._use_conv_reduction_block:
            # branchNet_gru_units -> in_channels -> channel_reduction_per_conv -> channel_seq
            # 12 -> in_channels = 288 -> 92 -> 288, 200, 112, 24
            # 24 -> in_channels = 576 -> 188 -> 576, 392, 208, 24
            # 32 -> in_channels = 768 -> 252 -> 768, 520, 272, 24
            in_channels = branchNet_gru_units * 2 * 12

            # The out_channels should be reduced to 12 in three steps
            first_out_channels = in_channels - (in_channels - 24) // 3
            second_out_channels = in_channels - 2 * (in_channels - 24) // 3
            third_out_channels = 24

            self._convolutional_reduction_block = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=first_out_channels,
                          kernel_size=conv_reduction_first_kernel_size, padding="same"),
                nn.BatchNorm1d(first_out_channels),
                nn.LeakyReLU(0.3),
                nn.Conv1d(in_channels=first_out_channels, out_channels=second_out_channels,
                          kernel_size=conv_reduction_second_kernel_size, padding="same"),
                nn.BatchNorm1d(second_out_channels),
                nn.LeakyReLU(0.3),
                nn.Conv1d(in_channels=second_out_channels, out_channels=third_out_channels,
                          kernel_size=conv_reduction_third_kernel_size, padding="same"),
                nn.LeakyReLU(0.3)
            )

        d_model = 2 * branchNet_gru_units * 12 if not self._use_conv_reduction_block else third_out_channels

        self._multi_head_contextual_attention = MultiHeadContextualAttention(d_model=d_model,
                                                        dropout=multi_branch_attention_dropout,
                                                        heads=multi_branch_heads,
                                                        # The following parameters are fixed for the paper
                                                        use_reduced_head_dims=True,
                                                        attention_activation_function="entmax15")


        self._batchNorm = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )

        # Input: d_model of the MH mechanism
        self._fcn = nn.Linear(in_features=d_model, out_features=num_classes)

        if apply_final_activation:
            self._final_activation = nn.Sigmoid() if multi_label_training else nn.LogSoftmax(dim=1)

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

        if self._use_conv_reduction_block:
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

        if self._apply_final_activation:
            x = self._final_activation(x)

        # Returns the multi-branch MACRO output as well as the 12 branchNet predictions
        return x, [single_lead_results[i][0] for i in range(12)]


if __name__ == "__main__":
    model = FinalModelMultiBranch(apply_final_activation=False,
                                  multi_label_training=True,
                                  branchNet_gru_units=12,
                                  use_conv_reduction_block=True,
                                  conv_reduction_first_kernel_size=3,
                                  conv_reduction_second_kernel_size=3,
                                  conv_reduction_third_kernel_size=3,
                                  branchNet_attention_dropout=0.2,
                                  branchNet_heads=6,
                                  branchNet_reduce_channels=False,
                                  multi_branch_attention_dropout=0.2,
                                  multi_branch_heads=24)
    summary(model, input_size=(2, 12, 15000), col_names=["input_size", "output_size", "kernel_size", "num_params"], depth=3)

    model_part = FinalModel(apply_final_activation=False,
                            multi_label_training=True,
                            input_channel=1,
                            num_classes=9,
                            # CNN-related parameters (almost all as in the default config)
                            pos_skip="all",
                            # GRU and MHA-related parameters
                            gru_units=12,
                            heads=6,
                            dropout_attention=0.2,
                            # The following parameters are fixed for the paper
                            use_reduced_head_dims=True,
                            attention_activation_function="entmax15",
                            # Multibranch-specific parameters
                            act_as_branch_net=True,
                            vary_channels_lighter_version=False)
    summary(model_part, input_size=(2, 1, 15000), col_names=["input_size", "output_size", "kernel_size", "num_params"],depth=1)
