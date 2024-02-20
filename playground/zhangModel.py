from resnet import resnet34
from torchinfo import summary

if __name__ == "__main__":
    net = resnet34(input_channels=12)
    summary(net, input_size=(2, 12, 15000), col_names=["input_size", "output_size", "num_params"])