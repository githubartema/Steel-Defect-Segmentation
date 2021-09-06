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
import config
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
import argparse

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

import segmentation_models_pytorch as smp

parser = argparse.ArgumentParser(description='Evaluate model')
parser.add_argument('-dir', type=str, default=None,
                    help='Pass a directory to evaluate the images in it', required=True)
parser.add_argument('-num_of_images', type=int, default=1,
                    help='Number of test image test.csv for segmentation', required=False)
parser.add_argument('-weights_dir', type=str, default=None,
                    help='Pass a weights directory', required=True)

args = parser.parse_args()
img_size = config.IMG_SIZE

print("Loading images from directory : ", args.dir)

path = args.dir
test = pd.read_csv(os.path.join(args.dir,'test.csv'))
test_ids = test['ImageId'].drop_duplicates().values
test_ids = test_ids[:args.num_of_images]

ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

test_dataset = utils.S_Dataset(df=test, datatype='test', folder=args.dir, img_ids=test_ids, transforms=utils.get_augmentation('valid'), preprocessing=utils.get_preprocessing(preprocessing_fn))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'

ACTIVATION = 'sigmoid'
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=1, 
    activation=ACTIVATION,
)

state_dict = torch.load(os.path.join(args.weights_dir, 'UnetEfficientNetB4_IoU_059.pth'), map_location=torch.device('cpu'))
model.load_state_dict(state_dict["model_state_dict"])
model.eval()

predicted_masks = []

for data_batch, _ in test_loader:
    data_batch = Variable(data_batch)        
    output_batch = model(data_batch)
    output_batch = output_batch.data.cpu().numpy()
    predicted_masks.append(output_batch)

for i in range(args.num_of_images):    
    image_name = test_ids[i]
    image = utils.get_img(image_name, folder=os.path.join(args.dir,'test_images'))
    predicted_mask = predicted_masks[i][0][0]
    utils.save_test(image, predicted_mask, image_name, args.dir)

print("Images are saved to result folder.")
