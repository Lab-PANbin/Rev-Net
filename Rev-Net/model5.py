import math
from typing import Sequence

import torch
import torch.nn as nn

from modules3 import (
    Conv2d,
    Conv2dZeros,
    ActNorm2d,
    InvertibleConv1x1,
    Permute2d,
    LinearZeros,
    SqueezeLayer,
    Split2d,
    gaussian_likelihood,
    gaussian_sample,
)
from utils import split_feature, uniform_binning_correction


def get_block(in_channels, image_channels, out_channels, hidden_channels):
    block1 = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
        nn.ReLU(inplace=False),
        #Conv2dZeros(hidden_channels, out_channels//2),
    )
    block2 = nn.Sequential(
        Conv2d(image_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
        nn.ReLU(inplace=False),
        #Conv2dZeros(hidden_channels, out_channels-out_channels//2),
    )
    block3 = nn.Sequential(
        Conv2d(2*hidden_channels, 4*hidden_channels),
        Conv2dZeros(4*hidden_channels, out_channels),
    )
    return block1,block2,block3

class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels,
        image_channels,
        hidden_channels,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )

        # 3. coupling
        if flow_coupling == "additive":
            self.block1, self.block2, self.block3 = get_block(in_channels // 2, image_channels, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.block1, self.block2, self.block3 = get_block(in_channels // 2, image_channels, in_channels, hidden_channels)

    def forward(self, input, image, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, image, logdet)
        else:
            return self.reverse_flow(input, image, logdet)

    def normal_flow(self, input, image, logdet):
        assert input.size(1) % 2 == 0

        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block1(z1)
        elif self.flow_coupling == "affine":
            h1=self.block1(z1)
            h2=self.block2(image)
            h3=torch.cat((h1,h2),1)
            h=self.block3(h3)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale+1e-6), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, image, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h1=self.block1(z1)
            h2=self.block2(image)
            h3=torch.cat((h1,h2),1)
            h=self.block3(h3)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale+1e-6), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(
        self,
        edm_shape,
        image_channels,
        hidden_channels,
        K,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        #image_shape=[100,100,224*4]
        H, W, C = edm_shape
        #C=C*4

        #for i in range(K):
            # 1. Squeeze
            #C, H, W = C * 4, H // 2, W // 2
            #self.layers.append(SqueezeLayer(factor=2))
            #self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
        for _ in range(K):
            self.layers.append(
                FlowStep(
                    in_channels=C,
                    image_channels=image_channels,
                    hidden_channels=hidden_channels,
                    actnorm_scale=actnorm_scale,
                    flow_permutation=flow_permutation,
                    flow_coupling=flow_coupling,
                    LU_decomposed=LU_decomposed,
                )
            )
            self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            #if i < L - 1:
            #    self.layers.append(Split2d(num_channels=C))
            #    self.output_shapes.append([-1, C // 2, H, W])
            #    C = C // 2

    def forward(self, input, image, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, image, temperature)
        else:
            return self.encode(input, image, logdet)

    def encode(self, z, image, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, image, logdet, reverse=False)
        return z, logdet

    def decode(self, z, image, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, image, logdet=0, reverse=True)
        return z, logdet

class UN(nn.Module):
    def __init__(
        self,
        image_shape,
        edm_shape,
        hidden_channels,
        K,
        P,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
    ):
        super().__init__()
        self.flow1 = FlowNet(
            edm_shape=edm_shape,
            image_channels=image_shape[1],
            hidden_channels=hidden_channels,
            K=K,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
        )
        self.flow2 = FlowNet(
            edm_shape=edm_shape,
            image_channels=image_shape[1],
            hidden_channels=hidden_channels,
            K=K,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
        )
        self.flow3 = FlowNet(
            edm_shape=edm_shape,
            image_channels=image_shape[1],
            hidden_channels=hidden_channels,
            K=K,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
        )
        self.P=P
        self.image_shape=image_shape
        self.abd=nn.Sequential(
            #nn.Conv2d(image_shape[1],image_shape[1]//2,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(image_shape[1],image_shape[1]//2,kernel_size=3,padding=1),
            nn.BatchNorm2d(image_shape[1]//2,affine=True),
            nn.ReLU(),
            nn.Conv2d(image_shape[1]//2,image_shape[1]//4,kernel_size=3,padding=1),
            nn.BatchNorm2d(image_shape[1]//4,affine=True),
            nn.ReLU(),
            nn.Conv2d(image_shape[1]//4,P,kernel_size=3,padding=1),
            nn.Softmax(dim=1)
        )
        self.ac2=nn.Sigmoid()
    def forward(self, image, rdn1, rdn2, rdn3):
        abd=self.abd(image)
        #print(abd.shape)
        abd=abd.reshape(-1,self.P).unsqueeze(dim=2)
        #print(rdn.shape)
        edm1, logdet1=self.flow1(rdn1,image)
        edm2, logdet2=self.flow2(rdn2,image)
        edm3, logdet3=self.flow3(rdn3,image)
        edm=torch.cat((edm1,edm2,edm3),dim=1)
        edm=self.ac2(edm).reshape(self.P*self.image_shape[1],-1).T.reshape(-1,self.image_shape[1],self.P)
        rec=torch.bmm(edm,abd)
        return abd,edm,rec,logdet1,logdet2,logdet3
