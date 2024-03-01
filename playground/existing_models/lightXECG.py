from torch import nn
from torchinfo import summary

import torch



class DSConv1d(nn.Module):
    def __init__(self,
        in_channels, out_channels,
        kernel_size, padding = 0, stride = 1,
    ):
        super(DSConv1d, self).__init__()
        self.dw_conv = nn.Conv1d(
            in_channels, in_channels,
            kernel_size = kernel_size, padding = padding, stride = stride,
            groups = in_channels,
            bias = False,
        )
        self.pw_conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size = 1,
            bias = False,
        )

    def forward(self,
        input,
    ):
        output = self.dw_conv(input)
        output = self.pw_conv(output)

        return output


class LightSEModule(nn.Module):
    def __init__(self,
        in_channels,
        reduction = 16,
    ):
        super(LightSEModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.s_conv = DSConv1d(
            in_channels, in_channels//reduction,
            kernel_size = 1,
        )
        self.act_fn = nn.ReLU()
        self.e_conv = DSConv1d(
            in_channels//reduction, in_channels,
            kernel_size = 1,
        )

    def forward(self,
        input,
    ):
        attention_scores = self.pool(input)

        attention_scores = self.s_conv(attention_scores)
        attention_scores = self.act_fn(attention_scores)
        attention_scores = self.e_conv(attention_scores)

        return input*torch.sigmoid(attention_scores)


class LightSEResBlock(nn.Module):
    def __init__(self,
        in_channels,
        downsample = False,
    ):
        super(LightSEResBlock, self).__init__()
        if downsample:
            self.out_channels = in_channels*2
            self.conv_1 = DSConv1d(
                in_channels, self.out_channels,
                kernel_size = 7, padding = 3, stride = 2,
            )
            self.identity = nn.Sequential(
                DSConv1d(
                    in_channels, self.out_channels,
                    kernel_size = 1, padding = 0, stride = 2,
                ),
                nn.BatchNorm1d(self.out_channels),
            )
        else:
            self.out_channels = in_channels
            self.conv_1 = DSConv1d(
                in_channels, self.out_channels,
                kernel_size = 7, padding = 3, stride = 1,
            )
            self.identity = nn.Identity()
        self.conv_2 = DSConv1d(
            self.out_channels, self.out_channels,
            kernel_size = 7, padding = 3, stride = 1,
        )

        self.convs = nn.Sequential(
            self.conv_1,
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            self.conv_2,
            nn.BatchNorm1d(self.out_channels),
            LightSEModule(self.out_channels),
        )
        self.act_fn = nn.ReLU()

    def forward(self,
        input,
    ):
        output = self.convs(input) + self.identity(input)
        output = self.act_fn(output)

        return output

class LightSEResNet18(nn.Module):
    def __init__(self,
        base_channels = 64,
    ):
        super(LightSEResNet18, self).__init__()
        self.bblock = LightSEResBlock
        self.stem = nn.Sequential(
            nn.Conv1d(
                1, base_channels,
                kernel_size = 15, padding = 7, stride = 2,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size = 3, padding = 1, stride = 2,
            ),
        )
        self.stage_0 = nn.Sequential(
            self.bblock(base_channels),
            self.bblock(base_channels),
        )

        self.stage_1 = nn.Sequential(
            self.bblock(base_channels*1, downsample = True),
            self.bblock(base_channels*2),
        )
        self.stage_2 = nn.Sequential(
            self.bblock(base_channels*2, downsample = True),
            self.bblock(base_channels*4),
        )
        self.stage_3 = nn.Sequential(
            self.bblock(base_channels*4, downsample = True),
            self.bblock(base_channels*8),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self,
        input,
    ):
        output = self.stem(input)
        output = self.stage_0(output)

        output = self.stage_1(output)
        output = self.stage_2(output)
        output = self.stage_3(output)

        output = self.pool(output)

        return output


class LightX3ECG(nn.Module):
    def __init__(self,
        base_channels = 64,
        num_classes = 1,
    ):
        super(LightX3ECG, self).__init__()
        self.backbone_0 = LightSEResNet18(base_channels)
        self.backbone_1 = LightSEResNet18(base_channels)
        self.backbone_2 = LightSEResNet18(base_channels)
        self.lw_attention = nn.Sequential(
            nn.Linear(
                base_channels*24, base_channels*8,
            ),
            nn.BatchNorm1d(base_channels*8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                base_channels*8, 3,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(
                base_channels*8, num_classes,
            ),
        )

    def forward(self,
        input,
        return_attention_scores = False,
    ):
        features_0 = self.backbone_0(input[:, 0, :].unsqueeze(1)).squeeze(2)
        features_1 = self.backbone_1(input[:, 1, :].unsqueeze(1)).squeeze(2)
        features_2 = self.backbone_2(input[:, 2, :].unsqueeze(1)).squeeze(2)
        attention_scores = torch.sigmoid(
            self.lw_attention(
                torch.cat(
                [
                    features_0,
                    features_1,
                    features_2,
                ],
                dim = 1,
                )
            )
        )
        merged_features = torch.sum(
            torch.stack(
            [
                features_0,
                features_1,
                features_2,
            ],
            dim = 1,
            )*attention_scores.unsqueeze(-1),
            dim = 1,
        )

        output = self.classifier(merged_features)

        if not return_attention_scores:
            return output
        else:
            return output, attention_scores


if __name__ == "__main__":
    net = LightX3ECG(num_classes=9)
    summary(net, input_size=(2, 3, 5000), col_names=["input_size", "output_size", "num_params"])