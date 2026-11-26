import torch
import torch.nn as nn
    
    

class VGG13(nn.Module):
    """
    VGG13 implementation for the CM2003 course lab, adapted for MNIST.
    
    This class implements a simplified VGG13 architecture, modified to work with
    MNIST input (1 channel, 28x28 images).

    Students should implement the network architecture following these steps:
    1. Create the features section with convolutional and pooling layers.
    2. Implement the adaptive average pooling layer.
    3. Create the classifier section with fully connected layers.
    4. Implement the forward method to define the data flow.

    Args:
        num_classes (int): Number of classes for the final classification layer. Default is 10.
        in_channels (int): Number of input channels. Default is 1 for MNIST.
    """

    def __init__(self, num_classes=10, in_channels=1, use_dropout_conv = False, dropout_prob = 0.4):
        super(VGG13, self).__init__()

        # If we want to use dropout on the convolutional blocks
        if use_dropout_conv:
            # ---CODE START HERE---
            # Step 1: Implement the features section
            self.features = nn.Sequential(#TODO
                # First block
                # Conv2d: in_channels -> 64, 3x3 kernel, 1 padding
                nn.Conv2d(in_channels,64,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),

                # Conv2d: 64 -> 64, 3x3 kernel, 1 padding
                nn.Conv2d(64,64,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # MaxPool2d: 2x2 kernel, 2 stride
                nn.MaxPool2d(2,stride=2),

                # Second block
                # Conv2d: 64 -> 128, 3x3 kernel, 1 padding
                nn.Conv2d(64,128,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # Conv2d: 128 -> 128, 3x3 kernel, 1 padding
                nn.Conv2d(128,128,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),

                # MaxPool2d: 2x2 kernel, 2 stride
                nn.MaxPool2d(2,stride=2),

                # Third block
                # Conv2d: 128 -> 256, 3x3 kernel, 1 padding
                nn.Conv2d(128,256,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # Conv2d: 256 -> 256, 3x3 kernel, 1 padding
                nn.Conv2d(256,256,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),

                # Add dropout
                nn.Dropout(dropout_prob),

                # MaxPool2d: 2x2 kernel, 2 stride
                nn.MaxPool2d(2,stride=2),

                # Fourth block
                # Conv2d: 256 -> 512, 3x3 kernel, 1 padding
                nn.Conv2d(256,512,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # Add dropout
                nn.Dropout(dropout_prob),

                # Conv2d: 512 -> 512, 3x3 kernel, 1 padding
                nn.Conv2d(512,512,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # MaxPool2d: 2x2 kernel, 2 stride
                nn.MaxPool2d(2,stride=2)
                )
            
        else:
            # ---CODE START HERE---
            # Step 1: Implement the features section
            self.features = nn.Sequential(#TODO
                # First block
                # Conv2d: in_channels -> 64, 3x3 kernel, 1 padding
                nn.Conv2d(in_channels,64,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # Conv2d: 64 -> 64, 3x3 kernel, 1 padding
                nn.Conv2d(64,64,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # MaxPool2d: 2x2 kernel, 2 stride
                nn.MaxPool2d(2,stride=2),
                # Second block
                # Conv2d: 64 -> 128, 3x3 kernel, 1 padding
                nn.Conv2d(64,128,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # Conv2d: 128 -> 128, 3x3 kernel, 1 padding
                nn.Conv2d(128,128,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # MaxPool2d: 2x2 kernel, 2 stride
                nn.MaxPool2d(2,stride=2),
                # Third block
                # Conv2d: 128 -> 256, 3x3 kernel, 1 padding
                nn.Conv2d(128,256,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # Conv2d: 256 -> 256, 3x3 kernel, 1 padding
                nn.Conv2d(256,256,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # MaxPool2d: 2x2 kernel, 2 stride
                nn.MaxPool2d(2,stride=2),
                # Fourth block
                # Conv2d: 256 -> 512, 3x3 kernel, 1 padding
                nn.Conv2d(256,512,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # Conv2d: 512 -> 512, 3x3 kernel, 1 padding
                nn.Conv2d(512,512,3,padding=1),
                # ReLU
                nn.ReLU(inplace=True),
                # MaxPool2d: 2x2 kernel, 2 stride
                nn.MaxPool2d(2,stride=2)
            )

        # Step 2: Implement the adaptive average pooling layer
        self.avgpool = None  # TODO: Use nn.AdaptiveAvgPool2d with output size (1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # Step 3: Implement the classifier section
        self.classifier = nn.Sequential(#TODO
            # Linear: 512 -> 4096
            nn.Linear(512,4096),
            # ReLU
            nn.ReLU(inplace=True),
            # Dropout
            nn.Dropout(),
            # Linear: 4096 -> 4096
            nn.Linear(4096,4096),
            # ReLU
            nn.ReLU(inplace=True),
            # Dropout
            nn.Dropout(),
            # Linear: 4096 -> num_classes
            nn.Linear(4096,num_classes)
        )
        # ---CODE END HERE---

    def forward(self, x):
        """
        Define the forward pass of the VGG13 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # ---CODE START HERE---
        # Step 4: Implement the forward method

        # Apply features
        x=self.features(x)
        # Apply avgpool
        x=self.avgpool(x)
        # Flatten the tensor
        x=torch.flatten(x,1)
        # Apply classifier
        x=self.classifier(x)
        # ---CODE END HERE---

        return x
