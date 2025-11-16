# model1.py
import torch
import torch.nn as nn
import torchvision.models as models

class Model1Pose(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()

        # SAME architecture used during training
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Replace FC layer (used in training)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_joints * 2)

    def forward(self, x):
        return self.model(x)
