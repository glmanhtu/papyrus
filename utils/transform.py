import numpy as np
import torchvision.transforms
import torchvision.transforms
from PIL import ImageOps
from imgaug import augmenters as iaa
from torchvision.transforms import transforms


def get_transforms(args):
    applying_percent = 0.3
    sometimes = lambda aug: iaa.Sometimes(applying_percent, aug)
    return torchvision.transforms.Compose([
        iaa.Sequential([
            sometimes(iaa.GaussianBlur(sigma=(0.0, 0.1))),
            sometimes(iaa.CoarseDropout(0.02, size_percent=0.5)),
            sometimes(iaa.LinearContrast((0.4, 1.6)))
        ]).augment_image,
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomApply([
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ], p=applying_percent),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def val_transforms(args):
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class UnNormalize(torchvision.transforms.Normalize):
    def __init__(self,mean,std,*args,**kwargs):
        new_mean = [-m/s for m,s in zip(mean,std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


def reverse_transform():
    return torchvision.transforms.Compose([
        UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torchvision.transforms.ToPILImage(),
        lambda image: ImageOps.invert(image)
    ])



