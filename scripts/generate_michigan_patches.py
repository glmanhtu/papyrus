import argparse
import os
import re

import cv2
import matplotlib
import numpy as np
import tqdm
from PIL import Image
from matplotlib import pyplot as plt, patches
from ml_engine.preprocessing.transforms import PadCenterCrop

# matplotlib.use("MacOSX")

parser = argparse.ArgumentParser('Pajigsaw patch generating script', add_help=False)
parser.add_argument('--data-path', required=True, type=str, help='path to dataset')
parser.add_argument('--output-path', required=True, type=str, help='path to output dataset')
parser.add_argument('--min-size-limit', type=int, default=112)
parser.add_argument('--patch-size', type=int, default=384)
args = parser.parse_args()


patch_size = args.patch_size
images = []
for root, dirs, files in os.walk(args.data_path):
    for file in files:
        if file.lower().endswith((".jpg", ".png")):
            images.append(os.path.join(root, file))

cropper = PadCenterCrop(patch_size, pad_if_needed=True, fill=255)
fragment_map = {}
for idx, image_path in enumerate(tqdm.tqdm(images)):
    with Image.open(image_path) as f:
        image = f.convert('RGB')

    if image.width * image.height < args.min_size_limit * args.min_size_limit:
        continue

    patch_dir = os.path.splitext(image_path)[0].replace(args.data_path, args.output_path)
    os.makedirs(patch_dir, exist_ok=True)
    # dpi = 80
    # im_visualize = np.asarray(image)
    # height, width, depth = im_visualize.shape
    # print(f'{width} x {height}')
    #
    # # What size does the figure need to be in inches to fit the image?
    # figsize = width / float(dpi), height / float(dpi)
    #
    # fig = plt.figure(figsize=figsize)
    # ax = fig.add_axes([0, 0, 1, 1])
    #
    # # Hide spines, ticks, etc.
    # ax.axis('off')
    #
    # # Display the image.
    # ax.imshow(image, cmap='gray')

    n_rows = max(round(image.height / patch_size), 1)
    n_cols = max(round(image.width / patch_size), 1)

    width = image.width // n_cols
    height = image.height // n_rows
    for i in range(n_rows):
        for j in range(n_cols):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            patch = image.crop(box)
            patch_name = f'{i}_{j}.jpg'
            patch.save(os.path.join(patch_dir, patch_name))
    #         rect = patches.Rectangle((box[0], box[1]), box[2] - box[0]-2, box[3] - box[1] - 2, linewidth=1,
    #                                  facecolor='green', alpha=0.5)
    #         ax.add_patch(rect)
    #
    # plt.show()
    # plt.close()