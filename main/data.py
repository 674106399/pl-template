import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os.path as osp
from config import cfg
import json

class RejDataset(Dataset):
    def __init__(self, json_file, img_folder, tfm=None):
        self.imgs = list(js.keys())
        self.labels = list(js.values())
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

    def __init__(self, tfms, json_file, data_dir='./'):
        super().__init__()
        self.batch_size = cfg.train_batch_size
        self.num_workers = cfg.num_thread
        self.tfms = tfms
        self.json_file = json_file
        self.data_dir = data_dir
        self.class_names = []
        self.num_classes = 335249
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, cfg.input_img_shape[0], cfg.input_img_shape[1])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        with open(self.json_file) as f:
            js = json.load(f)
        train_dataset = RejDataset(js['train'], osp.join(self.data_dir, cfg.dataset, 'train'), self.tfms['train'])
        val_dataset = RejDataset(js['val'], osp.join(self.data_dir, cfg.dataset, 'val'), self.tfms['val'])

        self.class_names = train_dataset.class_names
        self.num_classes = train_dataset.num_classes

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size)
