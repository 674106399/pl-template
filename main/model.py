# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import math
from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.preprocessing import load_classes
from nets.module import BackboneNet, Flatten, init_weights
from nets.loss import CircleLoss

class Model(pl.LightningModule):
    def __init__(self, backbone, feat_dim, num_classes):
        super().__init__()

        self.backbone_net = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d((1,1)),
        )
        convfeat_dim, expansion = self.get_convfeat_dim()
        
        self.convfeat_net = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(convfeat_dim * expansion * expansion)
        )
        self.fcfeat_net = nn.Sequential(
            nn.Linear(convfeat_dim * expansion * expansion, feat_dim),  # size / 16
            nn.BatchNorm1d(feat_dim)
        )
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.circleloss = CircleLoss(feat_dim, num_classes)

        # self.backbone_net.apply(init_weights)
        self.convfeat_net.apply(init_weights)
        self.fcfeat_net.apply(init_weights)
        self.classifier.apply(init_weights)

    def loss_func(self, preds, targets):
        return F.cross_entropy(preds, targets)

    def get_convfeat_dim(self):
        self.backbone_net.eval().to('cpu')
        dummy = torch.rand((1, 3, cfg.input_img_shape[0], cfg.input_img_shape[1]))
        dummy_res = self.backbone_net(dummy)
        res_shape = dummy_res.shape
        assert len(res_shape) == 4, 'res_shape.shape = ' + str(res_shape)
        return res_shape[1], res_shape[2]

    def forward(self, x):
        img_feat = self.backbone_net(x)
        conv_feat = self.convfeat_net(img_feat)
        fc_feat = self.fcfeat_net(conv_feat)
        out = self.classifier(fc_feat)
        return conv_feat, fc_feat, out
        # return conv_feat, fc_feat

    def acc(self, y_pred, y_true):
        accuracy = (y_pred.argmax(1) == y_true).float().mean()
        return accuracy

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        _, fc_feat, preds = self(inputs)
        # _, fc_feat = self(inputs)
        # loss = self.loss_func(preds, targets)
        loss = self.circleloss(fc_feat, targets) + self.loss_func(preds, targets)
        acc = self.acc(preds, targets)
        log = {
            'train_loss': loss
            'train_acc': acc
        }
        self.logger.experiment.log(log)
        return {'loss': loss, 
                'train_acc': acc,
                'progress_bar': log}

    def validation_step(self, batch, batch_idx):
        pass
        # inputs, targets = batch
        # _, _, preds = self(input_img)
        # loss = self.loss_func(preds, targets)
        # acc = self.acc(preds, targets)
        # log = {
        #     'val_loss': loss,
        #     'val_acc': acc
        # }
        # self.logger.experiment.log(log)
        # return {'loss': loss, 
        #         'val_acc': acc,
        #         'progress_bar': log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cfg.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--lr', type=float, default=0.0001)
        return parser

def get_model(num_classes, checkpoint_path=None):
    backbone_net = BackboneNet()
    model = Model(backbone_net, feat_dim=512, num_classes=num_classes)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['state_dict'])
    return model