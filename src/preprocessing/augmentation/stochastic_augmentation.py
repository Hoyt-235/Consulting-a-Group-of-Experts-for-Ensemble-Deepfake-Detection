import albumentations as A
import cv2
import os
import pprint
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import to_tensor
import numpy as np
from torchvision import transforms
from PIL import Image


dataAugmentation = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=0.0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=(-0.5, 0.5), p=1.0),
        ], p=1.0),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 19), p=1.0),
            A.Blur(blur_limit=(3, 19), p=1.0),
        ], p=1.0),

        A.OneOf([
            A.FancyPCA(alpha=0.5, p=1.0),
            A.CoarseDropout(num_holes_range=(1,3),
                            hole_height_range=(0.1,0.2),
                            hole_width_range=(0.1,0.2),
                            fill="random",
                            fill_mask=None,
                            p=1.0),
        ], p=1.0),

        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=(-90, 90),
                     border_mode=cv2.BORDER_REPLICATE,
                     p=1.0),
            A.VerticalFlip(p=1.0),
        ], p=1.0),

        A.OneOf([
            A.GaussNoise(std_range=(0.1,0.1), mean_range=(0,0), p=0.1),
            A.ImageCompression(quality_range=(1,10), p=0.9),
        ], p=1.0),
      A.ToTensorV2()
    ], p=1.0,  # apply the top-level montage with probability 1
       seed=1024)