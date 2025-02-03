# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict

# class ResNetFPN(nn.Module):
#     def __init__(self, initial_dim=128, block_dims=[128, 196, 256]):
#         super().__init__()
        
#         self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(initial_dim)
#         self.relu = nn.ReLU(inplace=True)
        
#         self.layer1 = self._make_layer(initial_dim, block_dims[0])
#         self.layer2 = self._make_layer(block_dims[0], block_dims[1], stride=2)
#         self.layer3 = self._make_layer(block_dims[1], block_dims[2], stride=2)
        
#         self.lateral3 = nn.Conv2d(block_dims[2], block_dims[2], 1)
#         self.lateral2 = nn.Conv2d(block_dims[1], block_dims[2], 1)
#         self.lateral1 = nn.Conv2d(block_dims[0], block_dims[2], 1)
        
#     def _make_layer(self, in_channels, out_channels, stride=1):
#         layers = []
#         layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
#         layers.append(nn.BatchNorm2d(out_channels))
#         layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
#         layers.append(nn.BatchNorm2d(out_channels))
#         layers.append(nn.ReLU(inplace=True))
#         return nn.Sequential(*layers)
    
#     def forward(self, x):

#         c1 = self.relu(self.bn1(self.conv1(x)))
#         c2 = self.layer1(c1)
#         c3 = self.layer2(c2)
#         c4 = self.layer3(c3)
        

#         p4 = self.lateral3(c4)
#         p3 = self.lateral2(c3) + F.interpolate(p4, size=c3.shape[-2:])
#         p2 = self.lateral1(c2) + F.interpolate(p3, size=c2.shape[-2:])
        
#         return [p2, p3, p4]

# class TransformerLayer(nn.Module):
#     def __init__(self, d_model, nhead, d_ffn):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead)
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
        
#     def forward(self, src, pos=None):
#         q = k = src if pos is None else src + pos
#         src2 = self.self_attn(q, k, src)[0]
#         src = src + src2
#         src = self.norm1(src)
        
#         src2 = self.linear2(F.relu(self.linear1(src)))
#         src = src + src2
#         src = self.norm2(src)
#         return src

# class LoFTR(nn.Module):
#     def __init__(self, config=None):
#         super().__init__()
#         if config is None:
#             config = {
#                 'backbone_type': 'ResNetFPN',
#                 'resolution': (8, 2),
#                 'fine_window_size': 5,
#                 'fine_concat_coarse_feat': True,
#                 'resnetfpn': {'initial_dim': 128, 'block_dims': [128, 196, 256]},
#                 'coarse': {
#                     'd_model': 256,
#                     'd_ffn': 256,
#                     'nhead': 8,
#                     'layer_names': ['self', 'cross'] * 4
#                 }
#             }
        
#         self.backbone = ResNetFPN(**config['resnetfpn'])

#         cfg = config['coarse']
#         self.pos_encoding = nn.Parameter(torch.randn(1, cfg['d_model'], 64, 64))
#         self.transformer_layers = nn.ModuleList([
#             TransformerLayer(cfg['d_model'], cfg['nhead'], cfg['d_ffn'])
#             for _ in range(len(cfg['layer_names']))
#         ])
        
#         self.coarse_matching = nn.Linear(cfg['d_model'], cfg['d_model'])
        
#     def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

#         feat0 = self.backbone(data['image0'])[-1]
#         feat1 = self.backbone(data['image1'])[-1]
        
#         b, c, h, w = feat0.shape
#         pos_encoding = F.interpolate(self.pos_encoding, size=(h, w))
#         feat0 = feat0 + pos_encoding
#         feat1 = feat1 + pos_encoding
        
#         feat0 = feat0.flatten(2).permute(2, 0, 1) 
#         feat1 = feat1.flatten(2).permute(2, 0, 1)
        
        
#         for layer in self.transformer_layers:
#             feat0 = layer(feat0)
#             feat1 = layer(feat1)
            
#         feat0 = feat0.permute(1, 2, 0).view(b, -1, h, w)
#         feat1 = feat1.permute(1, 2, 0).view(b, -1, h, w)
        
#         correlation = torch.einsum('bchw,bcij->bhwij', feat0, feat1)
#         scores = F.softmax(correlation / 0.1, dim=-1)
        
#         return {
#             'scores': scores,
#             'feat0': feat0,
#             'feat1': feat1
#         }


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class SimpleResNetFPN(nn.Module):
    def __init__(self, initial_dim=64, block_dims=[64, 128, 128]):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(initial_dim, block_dims[0])
        self.layer2 = self._make_layer(block_dims[0], block_dims[1], stride=2)
        self.layer3 = self._make_layer(block_dims[1], block_dims[2], stride=2)
        
        self.lateral3 = nn.Conv2d(block_dims[2], block_dims[2], 1)
        self.lateral2 = nn.Conv2d(block_dims[1], block_dims[2], 1)
        self.lateral1 = nn.Conv2d(block_dims[0], block_dims[2], 1)
        
    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        c1 = self.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        
        p4 = self.lateral3(c4)
        p3 = self.lateral2(c3) + F.interpolate(p4, size=c3.shape[-2:])
        p2 = self.lateral1(c2) + F.interpolate(p3, size=c2.shape[-2:])
        
        return [p2, p3, p4]

class SimpleTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, src, pos=None):
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, src)[0]
        src = src + src2
        src = self.norm1(src)
        
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)
        return src

class LoFTR(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {
                'backbone_type': 'SimpleResNetFPN',
                'resolution': (8, 2),
                'fine_window_size': 5,
                'fine_concat_coarse_feat': True,
                'resnetfpn': {'initial_dim': 64, 'block_dims': [64, 128, 128]},
                'coarse': {
                    'd_model': 128,
                    'd_ffn': 128,
                    'nhead': 4,
                    'layer_names': ['self', 'cross'] * 2
                }
            }
        
        self.backbone = SimpleResNetFPN(**config['resnetfpn'])

        cfg = config['coarse']
        self.pos_encoding = nn.Parameter(torch.randn(1, cfg['d_model'], 32, 32))
        self.transformer_layers = nn.ModuleList([
            SimpleTransformerLayer(cfg['d_model'], cfg['nhead'], cfg['d_ffn'])
            for _ in range(len(cfg['layer_names']))
        ])
        
        self.coarse_matching = nn.Linear(cfg['d_model'], cfg['d_model'])
        
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat0 = self.backbone(data['image0'])[-1]
        feat1 = self.backbone(data['image1'])[-1]
        
        b, c, h, w = feat0.shape
        pos_encoding = F.interpolate(self.pos_encoding, size=(h, w))
        feat0 = feat0 + pos_encoding
        feat1 = feat1 + pos_encoding
        
        feat0 = feat0.flatten(2).permute(2, 0, 1) 
        feat1 = feat1.flatten(2).permute(2, 0, 1)
        
        for layer in self.transformer_layers:
            feat0 = layer(feat0)
            feat1 = layer(feat1)
            
        feat0 = feat0.permute(1, 2, 0).view(b, -1, h, w)
        feat1 = feat1.permute(1, 2, 0).view(b, -1, h, w)
        
        correlation = torch.einsum('bchw,bcij->bhwij', feat0, feat1)
        scores = F.softmax(correlation / 0.1, dim=-1)
        
        return {
            'scores': scores,
            'feat0': feat0,
            'feat1': feat1
        }