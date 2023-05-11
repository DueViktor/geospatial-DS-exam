import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        num_input_channels=15,
        conv_filters1=4,
        conv_filters2=8,
        conv_filters3=4,
    ):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=4,
            kernel_size=3,
            padding=1,
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            padding=1,
        )
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            padding=1,
        )
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        c = self.conv1(x)
        x = self.relu1(c)
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)

        return x
