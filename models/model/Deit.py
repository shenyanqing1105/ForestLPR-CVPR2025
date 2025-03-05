# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
'''
single BEV version
'''
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg,
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = [
    'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_384',
]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class Flatten(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class GeMPooling(nn.Module):
    def __init__(self, feature_size, pool_size=7, init_norm=3.0, eps=1e-6, normalize=False, **kwargs):
        super(GeMPooling, self).__init__(**kwargs)
        self.feature_size = feature_size  # Final layer channel size, the pow calc at -1 axis
        self.pool_size = pool_size
        self.init_norm = init_norm
        self.p = torch.nn.Parameter(torch.ones(self.feature_size) * self.init_norm, requires_grad=True)
        self.p.data.fill_(init_norm)
        self.normalize = normalize
        self.avg_pooling = nn.AvgPool2d((self.pool_size, self.pool_size))
        self.eps = eps

    def forward(self, features):
        # filter invalid value: set minimum to 1e-6
        # features-> (B, H, W, C)
        features = features.clamp(min=self.eps).pow(self.p)
        features = features.permute((0, 3, 1, 2))
        features = self.avg_pooling(features)
        features = torch.squeeze(features)
        features = features.permute((0, 2, 3, 1))
        features = torch.pow(features, (1.0 / self.p))
        # unit vector
        if self.normalize:
            features = F.normalize(features, dim=-1, p=2)
        return features

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=True):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        if isinstance(x.size(-1), int):
            return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
        else:
            return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1).item())).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2).item(), x.size(-1).item())).pow(1./p)

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        inputc = kwargs.pop('inputc')
        self.agg = kwargs.pop('agg')
        self.level = kwargs.pop('level')
        levels = len(self.level)

        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pre = nn.Conv2d(in_channels=inputc, out_channels=3, kernel_size=1)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        if self.agg == 'cls':
            self.head = nn.Linear(self.embed_dim*levels, self.num_classes) if self.num_classes > 0 else nn.Identity()
            self.head_dist = nn.Linear(self.embed_dim*levels, self.num_classes) if self.num_classes > 0 else nn.Identity()
            self.head_dist.apply(self._init_weights)

        if self.agg == 'gem':
            self.gem = GeM()
            self.aggregation = nn.Sequential(L2Norm(), self.gem, Flatten(),
                                            nn.Linear(self.embed_dim*levels, self.num_classes),
                                            L2Norm())
        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.l2_norm = L2Norm()
        self.single = True

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.pre(x)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            features.append(self.norm(x))

        # x = self.norm(x)
        # return x[:, 0], x[:, 1] # CLS, DIS, B,_,C

        # stack different layers, depth=12
        return torch.stack(features, -3)

    def forward(self, x):
        x = self.forward_features(x) # tokenS
        x = x[:, self.level, :, :] #(B, L, 2+30*30, C) L=3 C=384 480/16
        x = x.permute(0,2,1,3).reshape(x.size(0), x.size(2), -1) #(B,2+30*30,C*L)
        # GeM
        # L = self.img_size[0] // self.patch_size
        # feat = x[:,2:,:].reshape(x.size(0),L,L,-1)
        # x = self.gem(feat)
        if self.agg=='gem':
            x = self.aggregation(x)
            return x

        # CLS+DIS
        elif self.agg=='cls':
            x, x_dist = self.l2_norm(x[:,0]), self.l2_norm(x[:,1]) #
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            return self.l2_norm((self.l2_norm(x) + self.l2_norm(x_dist)) / 2)

@register_model
def deit_small_distilled_patch16_224(pretrained=True, img_size=(224,224), num_classes =1000, inputc=1, depth=12, agg='gem', level=[1,3,5],**kwargs):
    model = DistilledVisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=384, depth=depth, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, inputc=inputc, agg=agg, level=level,**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        weight = checkpoint["model"]['pos_embed']
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
        new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
        matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
        resize = torchvision.transforms.Resize(new_size)
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
        checkpoint["model"]['pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)


        new_state_dict = model.state_dict()

        for name, param in checkpoint["model"].items():
            if name in new_state_dict and param.size() == new_state_dict[name].size():
                new_state_dict[name].copy_(param)
        model.load_state_dict(new_state_dict)
    return model

@register_model
def deit_base_distilled_patch16_384(pretrained=True, img_size=(384,384), num_classes =1000, inputc=1, depth=12, agg='gem', level=[1,3,5], **kwargs):
    model = DistilledVisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes, inputc=inputc, agg=agg, level=level, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )

        print(checkpoint["model"]['pos_embed'].shape)
        weight = checkpoint["model"]['pos_embed']
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
        new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
        matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
        resize = torchvision.transforms.Resize(new_size)
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
        print(new_matrix.shape)
        checkpoint["model"]['pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)
        print(checkpoint["model"]['pos_embed'].shape, model.pos_embed.shape)

        new_state_dict = model.state_dict()

        for name, param in checkpoint["model"].items():
            if name in new_state_dict and param.size() == new_state_dict[name].size():
                new_state_dict[name].copy_(param)
        model.load_state_dict(new_state_dict)
    return model
