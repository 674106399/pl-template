import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from config import cfg
class DataModule(pl.LightningDataModule):

    def __init__(self, tfms):
        super().__init__()
        self.batch_size = cfg.train_batch_size
        self.tfms = tfms

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, cfg.input_img_shape[0], cfg.input_img_shape[1])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_dataset = torchvision.datasets.ImageFolder('/home/jiangtao/rej_100w_imgfolder/', tfms['train'])
        val_dataset = torchvision.datasets.ImageFolder('/home/jiangtao/rej_100w_imgfolder/', tfms['val'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
