import math

import torch.nn as nn
from torchinfo import summary

from layers.CBAM import CBAM
from layers.LayerUtils import calc_same_padding_for_stride_one


class BasicBlock1dWithNorm(nn.Module):

    def __init__(self, in_channels, out_channels, mid_kernels_size, last_kernel_size, stride, down_sample, drop_out,
                 norm_type, norm_pos, skips_active=True, input_seq_len=None, cbam_active=False):
        """
        The stride is only applied in the last conv layer and can be used for size reduction
        If the resulting number of channel differs, it is done within the first Conv1D and kept for the further ones
        :param in_channels: Num of incoming channels
        :param out_channels: Num of resulting channels
        :param mid_kernels_size: Filter size for the first and second conv
        :param last_kernel_size: Filter size for the last conv
        :param stride: Stride to be applied for reducing the input size
        :param down_sample: Should be 'conv', 'max_pool' or 'avg_pool'
        :param drop_out: Dropout to be applied at the end of the block

        :param norm_type: BN, IN or LN
        :param norm_pos: All or last
        :param skips_active: If skip connection is enabled
        :param input_seq_len: Should be of an integer representing L=SeqLen (irrespective of the bs and the channels!)

        Information. The following restriction may be useful:
            Only odd filter sizes for all kernels with stride 1
            Otherwise for equal seq_lengths the original size can not be kept without one-sided padding
            -> Example: 72000 length, filter = 24, stride = 1
                        With p=11 -> 71999, with p=12 -> 72001 ==> Padding would have to be 2*11 + 1
        """
        # For Baseline: in_channels = out_channels = 12
        super().__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2, otherwise the sizes are too much reduced"
        assert norm_type != "LN" or input_seq_len is not None, "For LN, the input dimensions are required"

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._mid_kernels_size = mid_kernels_size
        self._last_kernel_size = last_kernel_size

        self._norm_type = norm_type
        self._norm_pos = norm_pos
        self._input_seq_len = input_seq_len

        self._cbam_active = cbam_active

        # If stride is 1 and the kernel_size uneven, an additional one-sided 0 is needed to keep/half dimension
        self._one_sided_padding = nn.ConstantPad1d((0, 1), 0)

        half_mid_kernel = calc_same_padding_for_stride_one(dilation=1, kernel_size=mid_kernels_size)  # k//2
        half_mid_kernel_minus_1 = (mid_kernels_size - 1) // 2
        self._half_mid_kernel_padding = nn.ConstantPad1d((half_mid_kernel, half_mid_kernel), 0)
        self._half_mid_kernel_padding_minus_1 = nn.ConstantPad1d((half_mid_kernel_minus_1, half_mid_kernel_minus_1), 0)

        half_last_kernel = calc_same_padding_for_stride_one(dilation=1, kernel_size=last_kernel_size)  # k//2
        half_last_kernel_minus_1 = (last_kernel_size - 1) // 2
        self._half_last_kernel_padding = nn.ConstantPad1d((half_last_kernel, half_last_kernel), 0)
        self._half_last_kernel_padding_minus_1 = nn.ConstantPad1d((half_last_kernel_minus_1, half_last_kernel_minus_1),
                                                                  0)

        self._conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=mid_kernels_size)
        if self._norm_pos == "all":
            if self._norm_type == "BN":
                self._norm1 = nn.BatchNorm1d(num_features=out_channels)
            elif self._norm_type == "IN":
                self._norm1 = nn.InstanceNorm1d(num_features=out_channels, affine=True)
            elif self._norm_type == "LN":
                self._norm1 = nn.LayerNorm([self._out_channels, self._input_seq_len])
        self._lrelu1 = nn.LeakyReLU(0.3)

        self._conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=mid_kernels_size)
        if self._norm_pos == "all":
            if self._norm_type == "BN":
                self._norm2 = nn.BatchNorm1d(num_features=out_channels)
            elif self._norm_type == "IN":
                self._norm2 = nn.InstanceNorm1d(num_features=out_channels, affine=True)
            elif self._norm_type == "LN":
                self._norm2 = nn.LayerNorm([self._out_channels, self._input_seq_len])
        self._lrelu2 = nn.LeakyReLU(0.3)

        self._conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=last_kernel_size,
                                stride=stride)
        # Before or after relu is handled during forward()
        if self._norm_pos == "all" or self._norm_pos == "last":
            if self._norm_type == "BN":
                self._norm3 = nn.BatchNorm1d(num_features=out_channels)
            elif self._norm_type == "IN":
                self._norm3 = nn.InstanceNorm1d(num_features=out_channels, affine=True)
            elif self._norm_type == "LN":
                self._norm3 = nn.LayerNorm([self._out_channels, math.ceil(self._input_seq_len/2)])
        self._lrelu3 = nn.LeakyReLU(0.3)
        self._dropout = nn.Dropout(drop_out)

        # self._batch_norm = nn.BatchNorm1d(num_features=out_channels, momentum=0.01)
        # self._instance_norm = nn.InstanceNorm1d(num_features=out_channels, affine=True)

        self._skips_active = skips_active
        self._downsample = None
        if self._skips_active:
            self._poooled_downsample = False
            if stride == 1 and in_channels == out_channels:
                # No downsampling needed
                self._downsample = None
            elif down_sample == 'conv':
                self._downsample = self._convolutional_downsample(stride=stride)
            elif down_sample == 'max_pool':
                self._downsample = self._max_pooled_downsample(stride=stride)
                self._poooled_downsample = True
            elif down_sample == 'avg_pool':
                self._downsample = self._avg_pooled_downsample(stride=stride)
                self._poooled_downsample = True

        if cbam_active:
            self.cbam = CBAM(out_channels, 6)

    def _convolutional_downsample(self, stride):
        # The block is potentially changing the channel amount
        # For stride ==1, down_sample is None, but for stride 2 it decreases the seq len by a factor of 2
        if self._norm_type == "BN":
            res_norm = nn.BatchNorm1d(num_features=self._out_channels)
        elif self._norm_type == "IN":
            res_norm = nn.InstanceNorm1d(num_features=self._out_channels, affine=True)
        elif self._norm_type == "LN":
            res_norm = nn.LayerNorm([self._out_channels, math.ceil(self._input_seq_len/stride)])
        downsample = nn.Sequential(
            nn.Conv1d(self._in_channels, self._out_channels, kernel_size=1, stride=stride, bias=False),
            res_norm
        )
        return downsample

    def _max_pooled_downsample(self, stride):
        if self._norm_type == "BN":
            res_norm = nn.BatchNorm1d(num_features=self._out_channels)
        elif self._norm_type == "IN":
            res_norm = nn.InstanceNorm1d(num_features=self._out_channels, affine=True)
        elif self._norm_type == "LN":
            res_norm = nn.LayerNorm([self._out_channels, math.ceil(self._input_seq_len/stride)])

        if self._in_channels == self._out_channels:
            downsample = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=stride),
                res_norm
            )
        else:
            downsample = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=stride),
                # Needed to align the channels before the addition
                nn.Conv1d(self._in_channels, self._out_channels, kernel_size=1, stride=1, bias=False),
                res_norm
            )
        return downsample

    def _avg_pooled_downsample(self, stride):
        if self._norm_type == "BN":
            res_norm = nn.BatchNorm1d(num_features=self._out_channels)
        elif self._norm_type == "IN":
            res_norm = nn.InstanceNorm1d(num_features=self._out_channels, affine=True)
        elif self._norm_type == "LN":
            res_norm = nn.LayerNorm([self._out_channels, math.ceil(self._input_seq_len/stride)])

        if self._in_channels == self._out_channels:
            downsample = nn.Sequential(
                nn.AvgPool1d(kernel_size=2, stride=stride),
                res_norm
            )
        else:
            downsample = nn.Sequential(
                nn.AvgPool1d(kernel_size=2, stride=stride),
                # Needed to align the channels before the addition
                nn.Conv1d(self._in_channels, self._out_channels, kernel_size=1, stride=1, bias=False),
                res_norm
            )
        return downsample

    def forward(self, x):
        # Handle padding explicit
        residual = x

        # Conv1 -----------------------------------------------------------------
        if self._mid_kernels_size % 2 == 0:
            x = self._one_sided_padding(x)
            x = self._half_mid_kernel_padding_minus_1(x)
        else:
            x = self._half_mid_kernel_padding(x)
        out = self._conv1(x)
        # Normalize here if "all"
        if self._norm_pos == "all":
            # BN or IN or LN
            out = self._norm1(out)
        out = self._lrelu1(out)

        # Conv2 -----------------------------------------------------------------
        if self._mid_kernels_size % 2 == 0:
            out = self._one_sided_padding(out)
            out = self._half_mid_kernel_padding_minus_1(out)
        else:
            out = self._half_mid_kernel_padding(out)
        out = self._conv2(out)
        # Normalize here if "all"
        if self._norm_pos == "all":
            # BN or IN or LN
            out = self._norm2(out)
        out = self._lrelu2(out)

        # Conv3 -----------------------------------------------------------------
        # Conv3 has stride 2
        if self._stride == 2:
            if self._last_kernel_size % 2 == 0 and out.shape[2] % 2 != 0:
                out = self._half_last_kernel_padding(out)
            else:
                out = self._half_last_kernel_padding_minus_1(out)
        else:
            if self._last_kernel_size % 2 == 0:
                out = self._one_sided_padding(out)
                out = self._half_last_kernel_padding_minus_1(out)
            else:
                out = self._half_last_kernel_padding(out)
        out = self._conv3(out)
        # Normalize here if "all" or "last"
        if self._norm_pos == "all" or self._norm_pos == "last":
            # BN or IN or LN
            out = self._norm3(out)

        # Residual -----------------------------------------------------------------
        if self._skips_active:
            if self._downsample is not None:
                if self._poooled_downsample:
                    # Stride is two and kernel size is two as well
                    if residual.shape[2] % 2 != 0:
                        residual = nn.ConstantPad1d((1, 1), 0)(residual)
                residual = self._downsample(residual)
            if self._cbam_active:
                out = self.cbam(out)
            out += residual

        # Import: Move the Leaky RELU part AFTER the addition as in ResNet!!!
        out = self._lrelu3(out)
        out = self._dropout(out)

        return out


if __name__ == "__main__":
    model = BasicBlock1dWithNorm(in_channels=12, out_channels=12, mid_kernels_size=3, last_kernel_size=44, stride=2,
                                 down_sample='conv', drop_out=0.2, norm_type="BN", norm_pos="all",
                                 input_dims=1125)
    summary(model, input_size=(5, 12, 1125), col_names=["input_size", "output_size", "num_params"])
