import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os.path as osp
from config import cfg
import json
from PIL import Image
from utils.preprocessing import Padding_resize
from torchvision.datasets import ImageFolder
import numpy as np

class SiameseDataset(Dataset):
    def __init__(self, json_file, img_folder, tfm=None):
        self.imgs = [osp.join(img_folder, x + '.jpg') for x in json_file.keys()]
        self.labels = list(json_file.values())
        self.tfm = tfm
        self.img_folder = img_folder

        self.class_names = list(set(self.labels))
        self.num_classes = len(self.class_names)
        # self.label2id = {}
        # for idx, label in enumerate(self.class_names):
        #     self.label2id[label] = idx

        self.label2imglist = {}
        for k in json_file:
            if json_file[k] not in self.label2imglist:
                self.label2imglist[json_file[k]] = set([k])
            else:
                self.label2imglist[json_file[k]].add(k)
        
    def __getitem__(self, index):
        img1 = Image.open(self.imgs[index]).convert('RGB')
        label1 = self.labels[index]
        w, h = img1.size
        if w != h:
            img1 = Padding_resize(img1)
        if self.tfm:
            img1 = self.tfm(img1)

        randnum = np.random.random()
        if randnum > (1.0 / cfg.train_batch_size):
            # 随机取一个负样例
            img2_idx = np.random.randint(0, self.imgs)
            img2 = Image.open(self.imgs[img2_idx]).convert('RGB')
            label2 = self.labels[img2_idx]
        else:
            # 取一个正样例
            img2_path = osp.join(self.img_folder, np.random.choice(self.label2imglist[label1]) + '.jpg')
            img2 = Image.open(img2_path)
            label2 = label1

        target = 1.0 if label1 == label2 else 0.0
        
        w, h = img2.size
        if w != h:
            img2 = Padding_resize(img2)
        if self.tfm:
            img2 = self.tfm(img2)
        return img1, img2, label1, target
    
    def __len__(self):
        return len(self.imgs)

class RejDataset(Dataset):
    def __init__(self, json_file, img_folder, tfm=None):
        self.imgs = [osp.join(img_folder, x + '.jpg') for x in json_file.keys()]
        self.labels = list(json_file.values())
        self.tfm = tfm

        self.class_names = list(set(self.labels))
        self.num_classes = len(self.class_names)
        self.label2id = {}
        for idx, label in enumerate(self.class_names):
            self.label2id[label] = idx
        
    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        label = self.labels[index]
        target = torch.tensor(self.label2id[label])

        w, h = img.size
        if w != h:
            img = Padding_resize(img)
        if self.tfm:
            img = self.tfm(img)
        return img, target
    
    def __len__(self):
        return len(self.imgs)

class DataModule(pl.LightningDataModule):

    def __init__(self, tfms, json_file=None, data_dir='./'):
        super().__init__()
        self.batch_size = cfg.train_batch_size
        self.num_workers = cfg.num_thread
        self.tfms = tfms
        self.json_file = json_file
        self.data_dir = data_dir

        if json_file:
            with open(self.json_file) as f:
                js = json.load(f)
            self.train_dataset = SiameseDataset(js['train'], osp.join(self.data_dir, cfg.dataset, 'train'), self.tfms['train'])
            # self.train_dataset = RejDataset(js['train'], osp.join(self.data_dir, cfg.dataset, 'train'), self.tfms['train'])
            # self.val_dataset = RejDataset(js['val'], osp.join(self.data_dir, cfg.dataset, 'val'), self.tfms['val'])
        else:
            self.train_dataset = ImageFolder(osp.join(data_dir, cfg.dataset, 'val'), self.tfms['train'])
            # self.val_dataset = ImageFolder(osp.join(data_dir, cfg.dataset, 'val'), self.tfms['val'])

        self.class_names = self.train_dataset.classes
        self.num_classes = len(self.class_names)
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, cfg.input_img_shape[0], cfg.input_img_shape[1])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size)
