import torch.cuda
import torch.nn as nn
from torchinfo import summary

from layers.LayerUtils import calc_same_padding_for_stride_one


class BasicBlock1dWithNormPreactivation(nn.Module):

    def __init__(self, in_channels, out_channels, mid_kernels_size, last_kernel_size, stride, down_sample, drop_out,
                 norm_type, norm_pos, norm_before_act=True):
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
        :param norm_type: Normalization to be used (BN or IN)
        :param norm_pos: Either "all" or "last" -> determines if normalization is used after each conv or only the last
        :param norm_before_act: Determines order (norm + L-ReLU vs L-ReLU + Norm)
        """
        # For Baseline: in_channels = out_channels = 12
        super().__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2, otherwise the sizes are too much reduced"
        assert norm_type == "BN" or norm_type == "IN", "Only BN and IN supported"

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._mid_kernels_size = mid_kernels_size
        self._last_kernel_size = last_kernel_size

        self._norm_type = norm_type
        self._norm_pos = norm_pos
        self._norm_before_act = norm_before_act

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
        # If applied before or after the lrelu is handled in forward method
        if self._norm_pos == "all":
            if self._norm_type == "BN":
                self._norm1 = nn.BatchNorm1d(num_features=out_channels)
            elif self._norm_type == "IN":
                self._norm1 = nn.InstanceNorm1d(num_features=out_channels, affine=True)
        self._lrelu1 = nn.LeakyReLU(0.3)

        self._conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=mid_kernels_size)
        # If applied before or after the lrelu is handled in forward method
        if self._norm_pos == "all":
            if self._norm_type == "BN":
                self._norm2 = nn.BatchNorm1d(num_features=out_channels)
            elif self._norm_type == "IN":
                self._norm2 = nn.InstanceNorm1d(num_features=out_channels, affine=True)
        self._lrelu2 = nn.LeakyReLU(0.3)

        self._conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=last_kernel_size,
                                stride=stride)
        # If applied before or after the lrelu is handled in forward method
        if self._norm_pos == "all" or self._norm_pos == "last":
            if self._norm_type == "BN":
                self._norm3 = nn.BatchNorm1d(num_features=out_channels)
            elif self._norm_type == "IN":
                self._norm3 = nn.InstanceNorm1d(num_features=out_channels, affine=True)
        self._lrelu3 = nn.LeakyReLU(0.3)

        self._dropout = nn.Dropout(drop_out)

        self._downsample = None
        self._poooled_downsample = False
        if stride == 1 and in_channels == out_channels:
            # No downsampling needed
            self._downsample = None
        elif down_sample == 'conv':
            self._downsample = self._convolutional_downsample(down_sample=stride)
        elif down_sample == 'max_pool':
            self._downsample = self._max_pooled_downsample(down_sample=stride)
            self._poooled_downsample = True
        elif down_sample == 'avg_pool':
            self._downsample = self._avg_pooled_downsample(down_sample=stride)
            self._poooled_downsample = True

    def _convolutional_downsample(self, down_sample):
        # The block is potentially changing the channel amount
        # For stride ==1, down_sample is None, but for stride 2 it decreases the seq len by a factor of 2
        downsample = nn.Conv1d(self._in_channels, self._out_channels, kernel_size=1, stride=down_sample, bias=False)
        return downsample

    def _max_pooled_downsample(self, down_sample):
        if self._in_channels == self._out_channels:
            downsample = nn.MaxPool1d(kernel_size=down_sample, stride=down_sample)
        else:
            downsample = nn.Sequential(
                nn.MaxPool1d(kernel_size=down_sample, stride=down_sample),
                # Needed to align the channels before the addition
                nn.Conv1d(self._in_channels, self._out_channels, kernel_size=1, stride=1, bias=False)
            )
        return downsample

    def _avg_pooled_downsample(self, down_sample):
        if self._in_channels == self._out_channels:
            downsample = nn.AvgPool1d(kernel_size=down_sample, stride=down_sample)
        else:
            downsample = nn.Sequential(
                nn.AvgPool1d(kernel_size=down_sample, stride=down_sample),
                # Needed to align the channels before the addition
                nn.Conv1d(self._in_channels, self._out_channels, kernel_size=1, stride=1, bias=False)
            )
        return downsample

    def forward(self, inputs):
        # Handle padding explicit
        x, residual = inputs

        # Conv1 -----------------------------------------------------------------
        if self._mid_kernels_size % 2 == 0:
            x = self._one_sided_padding(x)
            x = self._half_mid_kernel_padding_minus_1(x)
        else:
            x = self._half_mid_kernel_padding(x)
        out = self._conv1(x)
        # Normalize here if "all" and norm_before_act is True
        if self._norm_pos == "all" and self._norm_before_act:
            # BN or IN
            out = self._norm1(out)
        out = self._lrelu1(out)
        # Normalize here if "all" and norm_before_act is False
        if self._norm_pos == "all" and not self._norm_before_act:
            # BN or IN
            out = self._norm1(out)

        # Conv2 -----------------------------------------------------------------
        if self._mid_kernels_size % 2 == 0:
            out = self._one_sided_padding(out)
            out = self._half_mid_kernel_padding_minus_1(out)
        else:
            out = self._half_mid_kernel_padding(out)
        out = self._conv2(out)
        # Normalize here if "all" and norm_before_act is True
        if self._norm_pos == "all" and self._norm_before_act:
            # BN or IN
            out = self._norm2(out)
        out = self._lrelu2(out)
        if self._norm_pos == "all" and not self._norm_before_act:
            # BN or IN
            out = self._norm2(out)

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

        # Residual -----------------------------------------------------------------
        # In the preactivation design, the residual comes here!

        if self._downsample is not None:
            if self._poooled_downsample:
                # Stride is two and kernel size is two as well
                if residual.shape[2] % 2 != 0:
                    residual = nn.ConstantPad1d((1, 1), 0)(residual)
            residual = self._downsample(residual)

        out += residual
        residual = out

        # Rest of block (Norm + Activation + Dropout) --------------------------------------------------------

        # Normalize here if "all" or "last" and norm_before_act is True
        if (self._norm_pos == "all" or self._norm_pos == "last") and self._norm_before_act:
            # BN or IN
            out = self._norm3(out)
        out = self._lrelu3(out)
        if (self._norm_pos == "all" or self._norm_pos == "last") and not self._norm_before_act:
            # BN or IN
            out = self._norm3(out)

        out = self._dropout(out)

        return out, residual


if __name__ == "__main__":
    model = BasicBlock1dWithNormPreactivation(in_channels=12, out_channels=12, mid_kernels_size=3,
                                              last_kernel_size=44, stride=2,
                                              down_sample='conv', drop_out=0.2,
                                              norm_type="BN", norm_pos="all", norm_before_act=True)
    summary(model, input_size=(2, 2, 12, 1125), col_names=["input_size", "output_size", "num_params"])