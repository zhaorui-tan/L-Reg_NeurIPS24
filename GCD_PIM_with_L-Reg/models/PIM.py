import torch
import torch.nn as nn

import torch.nn.functional as F
from collections import OrderedDict




class MeanEncoder(nn.Module):
    """Identity function"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x


class VarianceEncoder(nn.Module):
    """Bias-only model with diagonal covariance"""
    def __init__(self, shape, init=0.1, channelwise=True, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps

        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = shape
        if channelwise:
            if len(shape) == 4:
                # [B, C, H, W]
                b_shape = (1, shape[1], 1, 1)
            elif len(shape ) == 3:
                # CLIP-ViT: [H*W+1, B, C]
                b_shape = (1, 1, shape[2])
            elif len(shape) == 2:
                # CLIP-ViT: [B, C]
                b_shape = (1, shape[1])
            else:
                raise ValueError()

        self.b = nn.Parameter(torch.full(b_shape, init))

    def forward(self, x):
        return F.softplus(self.b) + self.eps



class Affine(nn.Module):

    def __init__(self, num_features):
        super(Affine, self).__init__()
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(num_features, num_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(num_features, num_features)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(num_features, num_features)),
        ]))

        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear1.weight.data)
        nn.init.ones_(self.fc_gamma.linear1.bias.data)
        nn.init.zeros_(self.fc_beta.linear1.weight.data)
        nn.init.zeros_(self.fc_beta.linear1.bias.data)


    def forward(self, x, y=None, r=1.):
        bias = self.fc_beta(y)
        x = x + r * bias
        x = F.normalize(x, dim=1, p=2)
        return x


#################################
# This is modified PIM_partitioner
#################################


class PIM_partitioner(nn.Module):
    def __init__(self, num_features=512, num_classes=100, temp=25,  r=1.):
        super().__init__()
        self.partitioner = nn.Linear(num_features, num_classes, bias=False)
        self.temp = temp

        self.affine = Affine(num_features)
        self.r = r

    def forward(self, x, y=None, mb_lab_mask=None,):

        x = self.affine(x, x, r=self.r)
        out = self.partitioner(x)
        out = out * self.temp
        return x, out