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

matplotlib.use("TkAgg")

parser = argparse.ArgumentParser('Pajigsaw patch generating script', add_help=False)
parser.add_argument('--data-path', required=True, type=str, help='path to dataset')
parser.add_argument('--output-path', required=True, type=str, help='path to output dataset')
parser.add_argument('--min-size-limit', type=int, default=300)
parser.add_argument('--white-percentage-limit', type=float, default=0.75)
parser.add_argument('--patch-size', type=int, default=512)
args = parser.parse_args()


def compute_white_percentage(img, ref_size=224):
    gray = img.convert('L')
    if gray.width > ref_size:
        gray = gray.resize((ref_size, ref_size))
    gray = np.asarray(gray)
    white_pixel_count = np.sum(gray > 180)
    total_pixels = gray.shape[0] * gray.shape[1]
    return white_pixel_count / total_pixels


def extract_fragment_info(input_string, default_img_type='COLR'):
    # Define a regular expression pattern to match the initial numeric part
    pattern_1 = r'^([\w|-]+)([rRvV])_(\w+)$'
    pattern_2 = r'^([\w|-]+)_(\w+)$'

    match = re.search(pattern_1, input_string)

    if match:
        fragment_id = match.group(1)
        img_type = match.group(2)
        if img_type.lower() == 'r':
            img_type = 'COLR'
        else:
            img_type = 'COLV'
        remaining_text = match.group(3)

    else:
        match = re.match(pattern_2, input_string)
        if match:
            fragment_id = match.group(1)
            remaining_text = match.group(2)
            img_type = default_img_type
        else:
            raise Exception(f'Can not extract pattern from input string: {input_string}')
    return fragment_id, img_type, remaining_text

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

    if image.width < args.min_size_limit or image.height < args.min_size_limit:
        continue

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    fragment_id, img_type, _ = extract_fragment_info(image_name)

    patch_dir = os.path.join(args.output_path, fragment_id, f'{idx}_{img_type}-0')
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

    i = 0.
    while i * patch_size <= image.height:
        j = 0
        while j * patch_size <= image.width:
            box = (int(j * patch_size), int(i * patch_size),
                   min(int((j + 1) * patch_size), image.width), min(int((i + 1) * patch_size), image.height))
            patch = image.crop(box)
            patch = cropper(patch)
            white_percentage = compute_white_percentage(patch)
            if white_percentage < args.white_percentage_limit:
                patch_name = f'{image_name}_{round(i, 2)}_{round(j, 2)}.jpg'
                patch.save(os.path.join(patch_dir, patch_name))
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                         facecolor='green', alpha=0.5)
            # else:
            #     rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
            #                              facecolor='red', alpha=0.5)
            # ax.add_patch(rect)

            j += 1.01
        i += 1.01
    #
    # plt.show()
    # plt.close()