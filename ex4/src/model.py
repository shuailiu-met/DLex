from torch import nn


# ResBlock is the sequence inside the block , ResNet is the whole model including ResBlocks
# ResBlock corresponds to the ResBlock part in the Tab1, as one part of the whole chain
class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__(self)
        self.input_c = input_channels
        self.output_c = output_channels
        self.stride = stride
        self.input_tensor = None

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
        out = self.layer_sequence(input_tensor)
        conv1x1 = nn.Conv2d(in_channels=self.input_c, out_channels=self.output_c, kernel_size=1, stride=self.stride)
        out_11 = conv1x1(out)
        batch_norm = nn.BatchNorm2d(num_features=self.output_c)
        out_bn = batch_norm(out_11)
        out_end = out_bn + self.input_tensor
        relu_unit = nn.ReLU()
        out_end = relu_unit(out_end)
        return out_end


class ResNet(nn.Module):
    def __init__(self):
        super().__init__(self)
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
            nn.AvgPool2d(),
            nn.Flatten(),
            nn.Linear(in_features=512,out_features=2),
            nn.Sigmoid()
        )
