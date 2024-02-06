from abc import ABC

import torch.nn as nn
import torch
from torchinfo import summary

from base import BaseModel
from layers.BasicBlock1dWithNorm import BasicBlock1dWithNorm
from layers.BasicBlock1dWithNormPreActivationDesign import BasicBlock1dWithNormPreactivation
from layers.ContextualAttention import MultiHeadContextualAttention
from model.final_model_multibranch import FinalModelBranchNet


class FinalModelMultiBranchOld(BaseModel):

    def __init__(self,
                 multi_branch_gru_units=32,
                 multi_branch_heads=16,
                 vary_channels_lighter_version = True,
                 multi_label_training=True,
                 num_classes=9):
        """

        @return:
        """
        super().__init__()

        self._final_model_single_leads = [FinalModelBranchNet(
            gru_units=multi_branch_gru_units,
            heads=multi_branch_heads,
            vary_channels_lighter_version=vary_channels_lighter_version,
            multi_label_training=multi_label_training,
            input_channel=1) for _ in range(12)]

        self._test = nn.Sequential(*self._final_model_single_leads)

        # d_model = 2 * multi_branch_gru_units * 12
        self._multi_head_contextual_attention = MultiHeadContextualAttention(d_model=2 * multi_branch_gru_units * 12,
                                                                             dropout=0.3,
                                                                             heads=multi_branch_heads,
                                                                             discard_FC_before_MH=False)

        self._batchNorm = nn.Sequential(
            nn.BatchNorm1d(multi_branch_gru_units * 2 * 12),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )

        # Input: bidirectional GRU, twelve times -> multi_branch_gru_units * 2 * 12
        self._fcn = nn.Linear(in_features=multi_branch_gru_units * 2 * 12, out_features=num_classes)
        # apply final activation is false by default

    def forward(self, x):
        """

        @param x:  # x has shape [batch_size, 12, seq_len]
        @return:
        """
        # Format will be a list of length 12 containing of tuples of the following format
        # (final_output, biGRU_output) with
        # final_output: batch_size x 9
        # biGRU_output: batch_size x (seq_len/(2^5)) x (multi_branch_gru_units * 2)
        single_lead_results = [
            self._final_model_single_leads[i](x[:, i, None, :]) for i in range(12)
        ]
        # Concatenate the results of the single lead branches in channel dimension
        # Cat (BiGRU_1, ..., BiGRU_12)
        # BiGRU output has shape batch_size x (seq_len/(2^5)) x (multi_branch_gru_units * 2)
        # Hence, after concat, the shape is: batch_size x (seq_len/(2^5)) x (multi_branch_gru_units * 2 * 12)
        x = torch.cat([single_lead_results[i][1] for i in range(12)], dim=2)

        x = self._multi_head_contextual_attention(x)
        x = self._batchNorm(x)
        x = self._fcn(x)

        return x, [single_lead_results[i][0] for i in range(12)]


if __name__ == "__main__":
    model = FinalModelMultiBranchOld(
        multi_branch_gru_units=32,
        multi_branch_heads=16, )
    summary(model, input_size=(2, 12, 72000), col_names=["input_size", "output_size", "kernel_size", "num_params"], depth=5)

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
                                     gru_units=32,
                                     dropout_attention=0.3,  # MA model: 0.2,
                                     heads=16,
                                     discard_FC_before_MH=False)
    #summary(model_part, input_size=(2, 1, 72000), col_names=["input_size", "output_size", "kernel_size", "num_params"])
