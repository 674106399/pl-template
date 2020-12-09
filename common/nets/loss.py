import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from config import cfg
import math


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class CurricularFace(nn.Module):
    r"""Implement of CurricularFace (https://arxiv.org/pdf/2004.00288.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
        m: margin
        s: scale of outputs
    """
    def __init__(self, in_features, out_features, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output

class CircleLoss(nn.Module):
    def __init__(self, in_features, out_features, m=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.margin = m
        self.gamma = gamma
        self.class_num = out_features
        self.emdsize = in_features

        self.weight = nn.Parameter(torch.FloatTensor(self.class_num, self.emdsize))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        similarity_matrix = nn.functional.linear(nn.functional.normalize(input,p=2, dim=1, eps=1e-12), nn.functional.normalize(self.weight,p=2, dim=1, eps=1e-12))
        
        one_hot = torch.zeros_like(similarity_matrix)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = one_hot.type(dtype=torch.bool)
        #sp = torch.gather(similarity_matrix, dim=1, index=label.unsqueeze(1))
        sp = similarity_matrix[one_hot]
        mask = one_hot.logical_not()
        sn = similarity_matrix[mask]

        sp = sp.view(input.size()[0], -1)
        sn = sn.view(input.size()[0], -1)

        ap = torch.clamp_min(-sp.detach() + 1 + self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        output = torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)

        return output.mean()