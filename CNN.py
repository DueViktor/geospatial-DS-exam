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
            out_channels=conv_filters1,
            kernel_size=3,
            padding=1,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=conv_filters1,
            out_channels=conv_filters2,
            kernel_size=3,
            padding=1,
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=conv_filters2,
            out_channels=conv_filters3,
            kernel_size=3,
            padding=1,
        )
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(
            in_channels=conv_filters3, out_channels=1, kernel_size=3, padding=1
        )
        self.upsample = nn.Upsample(
            size=(256, 256), mode="bilinear", align_corners=False
        )

    def forward(self, x):
        c = self.conv1(x)
        x = self.relu1(c)
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.relu3(self.conv3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.upsample(x)

        return x
