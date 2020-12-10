import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from nets.layer import make_linear_layers, make_conv_layers, make_deconv_layers, make_upsample_layers
from nets.resnet import ResNetBackbone
import math

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        # self.net = ResNetBackbone(cfg.resnet_type)
        net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=False)
        self.net = nn.Sequential(*list(net.children())[:-2])

    def init_weights(self):
        self.net.init_weights()

    def forward(self, img):
        img_feat = self.net(img)
        return img_feat

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)