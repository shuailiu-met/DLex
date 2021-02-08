from torch import nn


# ResBlock is the sequence inside the block , ResNet is the whole model including ResBlocks
# ResBlock corresponds to the ResBlock part in the Tab1, as one part of the whole chain
class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super(ResBlock, self).__init__()
        self.input_c = input_channels
        self.output_c = output_channels
        self.stride = stride
        self.input_tensor = None
        self.conv1 = nn.Conv2d(self.input_c, self.output_c, kernel_size=1, stride=self.stride)
        self.batch_norm = nn.BatchNorm2d(self.output_c)
        self.relu = nn.ReLU()   # model for forward method

        self.layer_sequence = nn.Sequential(
            nn.Conv2d(in_channels=self.input_c, out_channels=self.output_c, kernel_size=3, stride=self.stride,
                      padding=1),
            nn.BatchNorm2d(num_features=self.output_c),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.output_c, out_channels=self.output_c, kernel_size=3, stride=1, padding=1),
            # for the second convlayer, the input size is already changed
            nn.BatchNorm2d(num_features=self.output_c),
            nn.ReLU()
        )

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_tensor = self.conv1(self.input_tensor)
        self.input_tensor = self.batch_norm(self.input_tensor)
        output_tensor = self.layer_sequence(input_tensor)
        output_tensor += self.input_tensor
        output_tensor = self.relu(output_tensor)
        return output_tensor


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
            # in channels RGB,3 channels for the first layer
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # now 64 channels
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            nn.AvgPool2d(kernel_size=(10, 10)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        out = self.seq(input_tensor)
        return out
