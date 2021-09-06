import os
import cv2
import collections

from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset

import albumentations as albu
from albumentations import pytorch as AT
from albumentations.pytorch import ToTensorV2

from catalyst.data import Augmentor
from catalyst import utils
from catalyst.contrib.data.cv.reader import ImageReader

def get_img(x, folder: str='train_images'):
    data_folder = folder
    image_path = os.path.join(data_folder, x)
    img = cv2.imread(image_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rle_decode(mask_rle: str = '', shape: tuple = (256, 1600)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def make_mask(df: pd.DataFrame, image_name: str='img.jpg', shape: tuple = (256, 1600)):
    encoded_masks = df.loc[df['ImageId'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 1), dtype=np.float32)

    mask = rle_decode(encoded_masks.values[0])
    masks[:,:,0] = mask
            
    return masks


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def save_test(image, mask, img_name, folder):
    fontsize = 14   
    f, ax = plt.subplots(2, 1, figsize=(30, 15))
    ax[0].imshow(cv2.resize(image, (config.IMG_SIZE[1], config.IMG_SIZE[0])))
    ax[1].imshow(cv2.resize(mask, (config.IMG_SIZE[1], config.IMG_SIZE[0])))
    f.savefig(os.path.join(folder, 'results', img_name + '.png'))
    
sigmoid = lambda x: 1 / (1 + np.exp(-x))

def get_augmentation(stage):
    if stage == 'train':
        transform = [
            albu.Resize(config.IMG_SIZE[0], config.IMG_SIZE[1]),
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
            albu.GridDistortion(p=0.5),
            albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        ]
    else:
        transform = [
        albu.Resize(config.IMG_SIZE[0], config.IMG_SIZE[1])
        ]

    return albu.Compose(transform)

def get_preprocessing(preprocessing_fn):
    transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(transform)

class S_Dataset(Dataset):
    def __init__(self, df, datatype, folder, img_ids, transforms = albu.Compose([albu.HorizontalFlip(),ToTensorV2()]), preprocessing=None):
        self.df = df
        if datatype != 'test':
            self.data_folder = os.path.join(folder,'train_images')
        else:
            self.data_folder = os.path.join(folder,'test_images')

        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)