import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, apply_final_activation, downsample):
        # For Baseline: in_channels = out_channels = 12
        super().__init__()
        self._apply_final_activation = apply_final_activation

        self._conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        self._lrelu1 = nn.LeakyReLU(0.3),
        self._conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        self._lrelu2 = nn.LeakyReLU(0.3),
        self._conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=24, stride=2, padding=11),
        self._lrelu3 = nn.LeakyReLU(0.3),
        self._dropout = nn.Dropout(0.2)

        if apply_final_activation:
            self._final_activation = nn.LeakyReLU(0.3)

        self._downsample = downsample

    def forward(self, x):
        residual = x
        out = self._conv1(x)
        out = self._lrelu1(out)
        out = self._conv2(out)
        out = self._lrelu2(out)
        out = self._conv3(out)
        out = self._lrelu3(out)
        out = self._dropout(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self._apply_final_activation:
            out = self._final_activation(out)
        return out






