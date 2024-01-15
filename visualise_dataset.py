import argparse
import logging

import albumentations as A
import cv2
import numpy as np
import torchvision.transforms
from ml_engine.preprocessing.transforms import ACompose

import transforms
from datasets.michigan_dataset import MichiganDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

parser = argparse.ArgumentParser('Pajigsaw testing script', add_help=False)
parser.add_argument('--data-path', required=True, type=str, help='path to dataset')
args = parser.parse_args()


patch_size = 384
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(patch_size, pad_if_needed=True, fill=(255, 255, 255)),
    ACompose([
        A.CoarseDropout(max_holes=32, min_holes=3, min_height=16, max_height=64, min_width=16, max_width=64,
                        fill_value=255, p=0.9),
    ]),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomApply([
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1),
    ], p=.5),
    transforms.GaussianBlur(p=0.5, radius_max=1),
    # transforms.Solarization(p=0.2),
    torchvision.transforms.RandomGrayscale(p=0.2),
])

train_dataset = MichiganDataset(args.data_path, MichiganDataset.Split.TRAIN, transforms=transform)
un_normaliser = torchvision.transforms.Compose([
    lambda x: np.asarray(x)
])
for img, label in train_dataset:
    first_img = un_normaliser(img)
    # if label[0] == 1:
    #     image = np.concatenate([first_img, second_img], axis=1)
    # elif label[2] == 1:
    #     image = np.concatenate([second_img, first_img], axis=1)
    # elif label[3] == 1:
    #     image = np.concatenate([second_img, first_img], axis=0)
    # elif label[1] == 1:
    #     image = np.concatenate([first_img, second_img], axis=0)
    #
    # else:
    #     image = np.concatenate([first_img, np.zeros_like(first_img), second_img], axis=0)

    # image = cv2.bitwise_not(image)
    cv2.imshow('image', cv2.cvtColor(first_img, cv2.COLOR_RGB2BGR))

    # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    cv2.waitKey(500)

    # cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()
