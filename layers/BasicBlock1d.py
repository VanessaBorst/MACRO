import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from layers.LayerUtils import calc_same_padding_for_stride_one


class BasicBlock1d(nn.Module):

    def __init__(self, in_channels, out_channels, mid_kernels_size, last_kernel_size, stride, down_sample, drop_out):
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

        Information. The following restriction may be useful:
            Only odd filter sizes for all kernels with stride 1
            Otherwise for equal seq_lengths the original size can not be kept without one-sided padding
            -> Example: 72000 length, filter = 24, stride = 1
                        With p=11 -> 71999, with p=12 -> 72001 ==> Padding would have to be 2*11 + 1
        """
        # For Baseline: in_channels = out_channels = 12
        super().__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2, otherwise the sizes are too much reduced"
        assert down_sample == "conv" or in_channels == out_channels, \
            "With different amount of in and out channels, pooling can not be used for aligning the residual tensor"

        self._in_channels = in_channels
        self._out_channels = out_channels

        # If stride is 1 and the kernel_size uneven, an additional one-sided 0 is needed to keep/half dimension
        one_sided_padding = nn.ConstantPad1d((0, 1), 0)
        if stride == 1 and mid_kernels_size % 2 == 0:
            self._conv1 = nn.Sequential(
                one_sided_padding,
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=mid_kernels_size,
                          padding=calc_same_padding_for_stride_one(dilation=1, kernel_size=mid_kernels_size)-1)
            )
            self._lrelu1 = nn.LeakyReLU(0.3)
            self._conv2 = nn.Sequential(
                one_sided_padding,
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=mid_kernels_size,
                          padding=calc_same_padding_for_stride_one(dilation=1, kernel_size=mid_kernels_size)-1)
            )
            self._lrelu2 = nn.LeakyReLU(0.3)
        else:
            self._conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=mid_kernels_size,
                                    padding=calc_same_padding_for_stride_one(dilation=1, kernel_size=mid_kernels_size))
            self._lrelu1 = nn.LeakyReLU(0.3)
            self._conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=mid_kernels_size,
                                    padding=calc_same_padding_for_stride_one(dilation=1, kernel_size=mid_kernels_size))
            self._lrelu2 = nn.LeakyReLU(0.3)

        # For stride 2, a distinction must be made between even and uneven kernel sizes as well
        if stride == 2:
            if last_kernel_size % 2 == 0: #and x ungerade
                padding_last = calc_same_padding_for_stride_one(dilation=1, kernel_size=last_kernel_size)
            else:
                padding_last = (last_kernel_size - 1) // 2
            self._conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=last_kernel_size,
                                    stride=stride, padding=padding_last)
        else:
            if last_kernel_size % 2 == 0:
                self._conv3 = nn.Sequential(
                    one_sided_padding,
                    nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=last_kernel_size,
                              padding=calc_same_padding_for_stride_one(dilation=1, kernel_size=last_kernel_size)-1)
                )
            else:
                self._conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                        kernel_size=last_kernel_size,
                                        padding=calc_same_padding_for_stride_one(dilation=1,
                                                                                 kernel_size=last_kernel_size))

        self._lrelu3 = nn.LeakyReLU(0.3)
        self._dropout = nn.Dropout(drop_out)

        self._downsample = None

        if stride == 1 and in_channels == out_channels:
            # No downsampling needed
            self._downsample = None
        elif down_sample == 'conv':
            self._downsample = self._convolutional_downsample(stride=stride)
        elif down_sample == 'max_pool':
            self._downsample = self._max_pooled_downsample()
        elif down_sample == 'avg_pool':
            self._downsample = self._avg_pooled_downsample()

    def _convolutional_downsample(self, stride):
        # The block is potentially changing the channel amount of 12
        # For stride ==1, down_sample is None, but for stride 2 it decreases the seq len by a factor of 2
        downsample = nn.Sequential(
            nn.Conv1d(self._in_channels, self._out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(self._out_channels)
        )
        return downsample

    def _max_pooled_downsample(self):
        # The block is keeping the channel amount of 12 but decreases the seq len by a factor of 2
        downsample = nn.MaxPool1d(kernel_size=2, stride=2)
        return downsample

    def _avg_pooled_downsample(self):
        # The block is keeping the channel amount of 12 but decreases the seq len by a factor of 2
        downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        return downsample

    def forward(self, x):
        residual = x
        out = self._conv1(x)
        out = self._lrelu1(out)
        out = self._conv2(out)
        out = self._lrelu2(out)
        out = self._conv3(out)
        out = self._lrelu3(out)
        out = self._dropout(out)
        if self._downsample is not None:
            residual = self._downsample(residual)
        out += residual
        return out


if __name__ == "__main__":
    model = BasicBlock1d(in_channels=12, out_channels=32, mid_kernels_size=5, last_kernel_size=21, stride=1,
                         down_sample='conv', drop_out=0.2)
    summary(model, input_size=(2, 12, 1125), col_names=["input_size", "output_size", "num_params"])
