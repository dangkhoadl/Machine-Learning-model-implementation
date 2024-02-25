import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from src.losses import SupConLoss
from typing import Optional, Dict, Union, Tuple
from dataclasses import dataclass

###### Projector ######
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

### Return dataclass
@dataclass
class EmbeddingsOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`):
            Classification hidden states before AMSoftmax.
        embeddings (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`):
            Utterance embeddings used for vector similarity-based retrieval.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    projected_embeddings: torch.FloatTensor = None # For plotting
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

###### Model ######
class Resnet_SCL_Config(PretrainedConfig):
    model_type = "resnet"

    def __init__(self, mode='pretrain', **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

class Resnet_SCL(PreTrainedModel):
    config_class = Resnet_SCL_Config

    def __init__(self, config):
        super().__init__(config)
        self.mode = config.mode

        ## Models
        # Load pretrained resnet50
        self.resnet_base = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_base.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.resnet_base.maxpool = Identity()
        self.resnet_base.fc = Identity()

        # Projector
        self.emb_projector = ProjectionHead(
            in_features=2048, hidden_features=2048, out_features=128)

        # Classifier
        self.classifier = nn.Linear(
            in_features=2048,
            out_features=config.num_classes)

        # Freeze/Unfreeze
        if self.mode == 'pretrain':
            # Unfreeze
            for param in self.resnet_base.parameters():
                param.requires_grad = True
            for param in self.emb_projector.parameters():
                param.requires_grad = True

            # Freeze
            for param in self.classifier.parameters():
                param.requires_grad = False
        else:
            # UnFreeze
            for param in self.classifier.parameters():
                param.requires_grad = True

            # Freeze
            for param in self.resnet_base.parameters():
                param.requires_grad = False
            for param in self.emb_projector.parameters():
                param.requires_grad = False

        # criterions
        self.criterion_pretrain = SupConLoss(
            temperature=0.8,
            base_temperature=0.07,
            contrast_mode='all')
        self.criterion_downstream = nn.CrossEntropyLoss()


    def forward(self, input_values, labels=None):
        bsz = input_values.shape[0]
        n_views = input_values.shape[1]
        feat_dim_1 = input_values.shape[2]
        feat_dim_2 = input_values.shape[3]
        feat_dim_3 = input_values.shape[4]

        # input_values.shape = (m, n_views, 3, 32, 32)
        # Concatenate views (bsz, views, 3, 32, 32) -> (bsz*views, 3, 32, 32)
        X_s = []
        for i in range(n_views):
            X_s.append(input_values[:, i, :, :, :])
        input_values = torch.cat(X_s, dim=0)

        # Forward: (bsz*views, 3, 32, 32) -> (bsz*views, 2048)
        pooled_output = self.resnet_base(input_values)

        # Project embeddings
        emb, logits = None, None
        if self.mode == 'pretrain':
            # Emb Projection: (views*bsz, out_dim) -> (views*bsz, 128)
            proj_emb = self.emb_projector(pooled_output)

            # L2 Normalization to prevent exploding gradients
            proj_emb = F.normalize(proj_emb, p=2, dim=-1)

            # Unstack: (3*bsz, 128) -> 3*(bsz, 128)
            f1, f2, f3 = torch.split(proj_emb, [bsz,bsz,bsz], dim=0)

            # Unstack pooled_output: (3*bsz, out_dim) -> 3*(bsz, out_dim)
            emb, _, _ = torch.split(pooled_output, [bsz,bsz,bsz], dim=0)
        else:
            logits = self.classifier(pooled_output)

        # Compute loss
        loss = None
        if labels is not None:
            if self.mode == 'pretrain':
                # Restack
                features = torch.cat([
                    f1.unsqueeze(1), f2.unsqueeze(1),
                    f3.unsqueeze(1)], dim=1)

                # Supervised Contrastive Loss mode
                loss = self.criterion_pretrain(features, labels)
            else:
                # Duplicate labels for each view
                labels_rep = []
                for i in range(n_views):
                    labels_rep.append(labels)
                labels_rep = torch.cat(labels_rep, dim=0).view(-1)

                # Compute loss
                logits = F.softmax(logits, dim=-1)
                loss = self.criterion_downstream(logits, labels_rep)

        return EmbeddingsOutput(
            loss=loss,
            logits=logits,
            projected_embeddings=emb, # (bz, out_dim) for plotting
        )
