import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from typing import Optional, Union

class Identity(nn.Module):
    """Gives output what it takes as input"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class LinearLayer(nn.Module):
    def __init__(self,
            in_features, out_features,
            use_bias=True, use_bn=False,
            **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn

        # Linear
        self.linear = nn.Linear(
            in_features=self.in_features, out_features=self.out_features,
            bias=self.use_bias and not self.use_bn)

        # Batch Norm
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        # Linear
        x = self.linear(x)

        # Batch norm
        if self.use_bn:
            x = self.bn(x)
        return x


class ProjectionHead(nn.Module):
    """Gives a linear or non-linear projection head according to the argument passed"""
    def __init__(self,
            in_features, hidden_features, out_features,
            head_type='nonlinear',
            **kwargs):
        super(ProjectionHead, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(
                in_features=self.in_features, out_features=self.out_features,
                use_bias=False, use_bn=True)

        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(
                    in_features=self.in_features, out_features=self.hidden_features,
                    use_bias=True, use_bn=True),
                nn.ReLU(),
                LinearLayer(
                    in_features=self.hidden_features,
                    out_features=self.out_features,
                    use_bias=False, use_bn=True))

    def forward(self, x):
        x = self.layers(x)
        return x

### Main ###
class PretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        ## Load pretrained resnet50
        self.pretrained = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT)
        self.pretrained.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.pretrained.maxpool = Identity()
        self.pretrained.fc = Identity()

        # Unfreeze pretrained
        for p in self.pretrained.parameters():
            p.requires_grad = True

        ## Projector
        self.projector = ProjectionHead(
            in_features=2048, hidden_features=2048, out_features=128)

    def forward(self, x):
        # (m, 3, 32, 32) -> (m, 2048)
        out = self.pretrained(x)

        # (m, 2048) -> (m, 128)
        xp = self.projector(torch.squeeze(out))
        return xp

class DownstreamModel(nn.Module):
    def __init__(self, premodel:Optional[Union[str, nn.Module]], num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        ## Load pretrained
        if premodel == 'resnet50':
            self.premodel = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT)
            self.premodel.conv1 = nn.Conv2d(
                in_channels=3, out_channels=64,
                kernel_size=(3, 3), stride=(1, 1), bias=False)
            self.premodel.maxpool = Identity()
            self.premodel.fc = Identity()

        else:
            self.premodel = premodel

        # Freeze pretrained
        for p in self.premodel.parameters():
            p.requires_grad = False

        # Freeze projector if exists
        if hasattr(self.premodel, 'projector'):
            for p in self.premodel.projector.parameters():
                p.requires_grad = False

        # Classifier
        self.fc = nn.Linear(
            in_features=2048,
            out_features=self.num_classes)

    def forward(self,x):
        # (m, 3, 32, 32) -> (m, 2048)
        if hasattr(self.premodel, 'pretrained'):
            out = self.premodel.pretrained(x)
        else:
            out = self.premodel(x)

        # (m, 2048) -> (m, num_classes)
        out = self.fc(out)
        return F.softmax(out, dim=-1)

