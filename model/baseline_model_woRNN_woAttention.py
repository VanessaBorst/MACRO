import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from layers.contextualAttention import ContextualAttention
from torchinfo import summary

from utils import plot_record_from_np_array


class BaselineModelWoRnnWoAttention(BaseModel):
    def __init__(self, apply_final_activation, multi_label_training, num_classes=9, num_cnn_blocks=5):
        """
        :param apply_final_activation: whether the Sigmoid(sl) or the LogSoftmax(ml) should be applied at the end
        :param multi_label_training: if true, Sigmoid is used as final activation, else the LogSoftmax
        :param num_classes: Num of classes to classify
        :param num_cnn_blocks: Num of CNN blocks to use
        TODO functionality for num_cnn_blocks
        """
        super().__init__()
        self._apply_final_activation = apply_final_activation
        self._conv_block1 = nn.Sequential(
            # Keras Code -> Input shape (7200, 12) -> in_channel = 12
            # x = Convolution1D(12, 3, padding='same')(main_input) -> Stride 1 as default
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=24, stride=2, padding=11),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )
        self._conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=24, stride=2, padding=11),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )
        self._conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=24, stride=2, padding=11),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )
        self._conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=24, stride=2, padding=11),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )
        # Last block has a convolutional pooling with kernel size 48!
        self._conv_block5 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=48, stride=2, padding=23),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )

        self._batchNorm = nn.Sequential(
            # The batch normalization layer has 12*2=24 trainable and 12*2=24 non-trainable parameters
            nn.BatchNorm1d(12),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2)
        )

        self._fcn = nn.Linear(in_features=2250*12, out_features=num_classes)

        if apply_final_activation:
            self._final_activation = nn.Sigmoid() if multi_label_training else nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Plot the first sample from the batch
        # plot_record_from_np_array(x[1].detach().numpy())
        x = self._conv_block1(x)
        # plot_record_from_np_array(x[1].detach().numpy())
        x = self._conv_block2(x)
        # plot_record_from_np_array(x[1].detach().numpy())
        x = self._conv_block3(x)
        # plot_record_from_np_array(x[1].detach().numpy())
        x = self._conv_block4(x)
        # plot_record_from_np_array(x[1].detach().numpy())
        x = self._conv_block5(x)
        # plot_record_from_np_array(x[1].detach().numpy())
        # should be the same as x.flatten(start_dim=1)
        x = self._batchNorm(x)
        x = x.reshape(x.size(0), -1)
        x = self._fcn(x)
        return self._final_activation(x) if self._apply_final_activation else x


if __name__ == "__main__":
    model = BaselineModelWoRnnWoAttention()
    summary(model, input_size=(2, 12, 72000), col_names=["input_size", "output_size", "num_params"])
