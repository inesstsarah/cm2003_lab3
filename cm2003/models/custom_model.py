import torch
import torch.nn as nn
# Because the 

class CustomModel(nn.Module):
    """
    Your own model for the Model playground notebook (optional).

    Design any architecture you like, for example by combining ideas from AlexNet, VGG and ResNet
    (batch normalization, dropout, residual connections, ...). Use it in the notebook with:

        from cm2003.models.custom_model import CustomModel
        model = CustomModel(num_classes=7, in_channels=3)
    """

    def __init__(self, num_classes=7, in_channels=3):
        super().__init__()
        # ---CODE START HERE---

        # ---CODE END HERE---

    def forward(self, x):
        # ---CODE START HERE---

        # ---CODE END HERE---
        return x
