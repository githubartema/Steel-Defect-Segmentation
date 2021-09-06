import os
import cv2
import collections
import time 

import tqdm
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from functools import partial
train_on_gpu = True
import argparse
import config 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import albumentations as albu
from albumentations import pytorch as AT
from albumentations.pytorch import ToTensorV2

from catalyst.data import Augmentor
from catalyst.contrib.data.cv.reader import ImageReader
from catalyst.contrib.data.reader import ScalarReader, ReaderCompose, LambdaReader
from catalyst.dl import SupervisedRunner, SchedulerCallback, TensorboardLogger
from catalyst.callbacks.metrics.segmentation import DiceCallback, IOUCallback, TrevskyCallback
from catalyst.callbacks.checkpoint import CheckpointCallback
from catalyst.callbacks.misc import EarlyStoppingCallback
from catalyst.callbacks import EarlyStoppingCallback, CheckpointCallback

from utils import utils, losses

import segmentation_models_pytorch as smp


parser = argparse.ArgumentParser(description='Training stage')
parser.add_argument('-dir', type=str, default=None,
                    help='Pass a directory to the train images', required=True)
parser.add_argument('-encoder', type=str, default='efficientnet-b3',
                    help='Backbone to use as encoder for UNet', required=False)
parser.add_argument('-batch_size', type=int, default=8,
                    help='Batch size for training', required=False)
parser.add_argument('-num_workers', type=int, default=0,
                    help='Number of workers for training', required=False)


args = parser.parse_args()

img_size = config.IMG_SIZE
batch_size = args.batch_size
path = args.dir 

train = pd.read_csv(os.path.join(args.dir,'train_balanced.csv'))
train = train.drop_duplicates(subset=['ImageId']).reset_index(drop=True)
id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'ImageId'].value_counts().\
reset_index().rename(columns={'index': 'img_id', 'ImageId': 'count'})

train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1, shuffle=True)

ENCODER = args.encoder 
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

ACTIVATION = 'sigmoid'
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=1, 
    activation=ACTIVATION,
)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

num_workers = args.num_workers
bs = args.batch_size

train_dataset = utils.S_Dataset(df=train, folder=args.dir, datatype='train', img_ids=train_ids, transforms = utils.get_augmentation('train'), preprocessing=utils.get_preprocessing(preprocessing_fn))
valid_dataset = utils.S_Dataset(df=train, folder=args.dir, datatype='val', img_ids=valid_ids, transforms = utils.get_augmentation('val'),  preprocessing=utils.get_preprocessing(preprocessing_fn))

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

num_epochs = 40

optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-3}, 
    {'params': model.decoder.parameters(), 'lr': 1e-2}, 
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.15, patience=2
    )

criterion = losses.DiceBCELoss()

runner = SupervisedRunner(
    input_key='features', 
    output_key='scores', 
    target_key='targets',
    loss_key='loss'
)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    callbacks=[
               IOUCallback(input_key="scores",target_key="targets"),
               TrevskyCallback(input_key="scores", target_key="targets", alpha=0.1),
               SchedulerCallback(loader_key='valid', metric_key='loss'),
               EarlyStoppingCallback(loader_key='valid', metric_key='loss', 
                                     minimize=True,patience=3, min_delta=0.001),
               CheckpointCallback(
                   logdir='checkpoint/', 
                   loader_key='valid', metric_key="loss", save_n_best=2, 
                   minimize=True)
               
               ],
    loaders=loaders,
    num_epochs=num_epochs,
    verbose=True,
    loggers={'tensorboard': TensorboardLogger(logdir='logdir/tensorboard')},
    logdir='logdir/tensorboard'
)

print('Weights have been saved to checkpoint library.')
