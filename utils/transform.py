import torchvision.transforms
import torchvision.transforms
from PIL import ImageOps
from torchvision.transforms import transforms


def get_transforms():
    applying_percent = 0.3
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomApply([
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ], p=applying_percent),
        torchvision.transforms.RandomHorizontalFlip(),
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



