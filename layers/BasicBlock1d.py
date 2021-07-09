import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class BasicBlock1d(nn.Module):

    def __init__(self, in_channels, out_channels, last_kernel_size, down_sample):
        # For Baseline: in_channels = out_channels = 12
        super().__init__()

        self._conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self._lrelu1 = nn.LeakyReLU(0.3)
        self._conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self._lrelu2 = nn.LeakyReLU(0.3)
        self._conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=last_kernel_size,
                                stride=2, padding=int(last_kernel_size/2)-1)
        self._lrelu3 = nn.LeakyReLU(0.3)
        self._dropout = nn.Dropout(0.2)

        self._downsample = self._convolutional_downsample() if down_sample == 'conv' else self._pooled_downsample()

    def _convolutional_downsample(self):
        in_planes = self._conv1.in_channels
        out_planes = self._conv3.out_channels
        # The block is keeping the channel amount of 12 but decreases the seq len by a factor of 2
        downsample = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(out_planes)
        )
        return downsample

    def _pooled_downsample(self):
        # The block is keeping the channel amount of 12 but decreases the seq len by a factor of 2
        downsample = nn.MaxPool1d(kernel_size=2, stride=2)
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
            residual = self._downsample(x)
        out += residual
        return out


if __name__ == "__main__":
    model = BasicBlock1d(in_channels=12, out_channels=12, last_kernel_size=48, down_sample="conv")
    summary(model, input_size=(2, 12, 72000), col_names=["input_size", "output_size", "num_params"])


