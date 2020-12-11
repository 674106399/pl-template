import os
from argparse import ArgumentParser
from config import cfg
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from data import DataModule
from model import Model, get_model
from utils.preprocessing import tfms

def main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = DataModule(tfms, '/home/jiangtao/rej_100w_20201209.json', '/home/jiangtao/')
    print('num_classes:', dm.num_classes)
    # ------------
    # model
    # ------------
    model = get_model(dm.num_classes)

    checkpoint_callback = ModelCheckpoint(
        monitor='v_loss',
        filename='rej-{epoch:02d}-{v_loss:.3f}-{v_recall:.3f}',
        mode='min',
    )


    # ------------
    # training
    # ------------
    wandb_logger = WandbLogger(offline=True)
    trainer = pl.Trainer.from_argparse_args(
        # training settings
        args,
        distributed_backend='ddp',
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        gpus=[1,2,5,6],
        # profiler=True,

        # debug
        # fast_dev_run=True,
        # limit_train_batches=10,
        # limit_val_batches=0.0,

        # training tricks
        # accumulate_grad_batches=10,
        # early_stop_callback=early_stop_callback,
        max_epochs=cfg.end_epoch,
        precision=16, 
        benchmark=True
    )
    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    # trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()