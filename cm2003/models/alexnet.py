import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    AlexNet implementation for the CM2003 course lab.
    
    This class implements the AlexNet architecture as described in the paper:
    "ImageNet Classification with Deep Convolutional Neural Networks" by Krizhevsky et al.

    Students should implement the network architecture following these steps:
    1. Create the features section with convolutional and pooling layers.
    2. Implement the adaptive average pooling layer.
    3. Create the classifier section with fully connected layers.
    4. Implement the forward method to define the data flow.

    Args:
        num_classes (int): Number of classes for the final classification layer. Default is 10.
        in_channels (int): Number of input channels. Default is 1.
    """

    def __init__(self, num_classes=10, in_channels=1):
        super(AlexNet, self).__init__()
    
        # ---CODE START HERE---
        # Step 1: Implement the features section for example in a sequential manner

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32,kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2(32, 64, kernel_size = 3, stride = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            # Conv1: 3x3 kernel, 1 stride, 1 padding, 32 output channels
            #  relu
            # MaxPool1: 2x2 kernel, 2 stride
            # Conv2: 3x3 kernel, 1 stride, 1 padding, 64 output channels
            # relu
            # MaxPool2: 2x2 kernel, 2 stride
            # Conv3: 3x3 kernel, 1 stride, 1 padding, 128 output channels
            # relu
        )

        # Step 2: Implement the adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((6,6)) # Not sure what size actually TODO

        # Step 3: Implement the classifier section
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3 * 256),
            nn.RelU(inplace=True),
            nn.Linear(256,num_classes),
            # Linear (128 * 3 * 3, 256)
            # relu
            # Linear (256, num classes)          
        )
        # ---CODE END HERE---

    def forward(self, x):
        """
        Define the forward pass of the AlexNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # ---CODE START HERE---
        # Step 4: Implement the forward method
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # ---CODE END HERE---
        return x


