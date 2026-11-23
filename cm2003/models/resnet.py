import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    """
    ResNet block implementation for the CM2003 course lab.
    
    This class implements a single ResNet block as described in the paper:
    "Deep Residual Learning for Image Recognition" by He et al.

    Students should implement the block architecture following these steps:
    1. Create the convolutional layers with batch normalization.
    2. Implement the skip connection (identity_downsample).
    3. Implement the forward method to define the data flow.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for the first convolutional layer. Default is 1.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.expansion = 4
        self.stride = stride

        #--CODE START--
        # Step 1: Implement the convolutional layers with batch normalization
        # Conv1: 1x1 kernel, 1 stride, no padding, out_channels output channels
        # BatchNorm1
        # Conv2: 3x3 kernel, stride as input, 1 padding, out_channels output channels
        # BatchNorm2
        # Conv3: 1x1 kernel, 1 stride, no padding, out_channels * expansion output channels
        # BatchNorm3
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size = 3, padding = 1) # TODO: Check if this is correct
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size = 1)
        self.batchnorm3 = nn.BatchNorm2d(out_channels*self.expansion)




        # Step 2: Implement the skip connection
        self.identity_downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            # Downsample the input to match the output dimensions
            self.identity_downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels * self.expansion,
                    kernel_size = 1, 
                    stride = self.stride,
                    bias = False

                ),
                nn.BatchNorm2d(out_channels*self.expansion)

            )
            pass
  

        self.relu = nn.ReLU()
        #--CODE END--

    def forward(self, x):
        #--CODE START--
        #out = x
        # Step 3: Implement the forward method
        # Save input as identity
        # Apply Conv1, BatchNorm1, and ReLU
        # Apply Conv2, BatchNorm2, and ReLU
        # Apply Conv3 and BatchNorm3
        # Apply identity_downsample if it exists
        # Add identity to the output
        # Apply final ReLU
        identity = x
        x = self.conv1(x)
        #print(x.shape)
        x = self.batchnorm1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.batchnorm2(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.batchnorm3(x)
        #print(x.shape)

        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x = x +  identity
        x = self.relu(x)
        out = x
        


        #--CODE END--

        return out

class ResNet50(nn.Module):
    """
    ResNet-50 implementation for the CM2003 course lab.
    
    This class implements the ResNet-50 architecture as described in the paper:
    "Deep Residual Learning for Image Recognition" by He et al.

    Args:
        num_classes (int): Number of classes for the final classification layer. Default is 1000.
        in_channels (int): Number of input channels. Default is 3.
    """

    def __init__(self, num_classes=1000, in_channels=3):
        super(ResNet50, self).__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * 4
        for _ in range(1, blocks):
            layers.append(ResNetBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

