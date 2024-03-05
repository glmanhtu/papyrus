import argparse
import glob
import os.path
import re

import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser('Pajigsaw patch generating script', add_help=False)
parser.add_argument('--data-path', required=True, type=str, help='path to dataset')
parser.add_argument('--output-path', required=True, type=str, help='path to output dataset')
args = parser.parse_args()

def white_to_transparency(img):
    x = np.asarray(img.convert('RGBA')).copy()

    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)


images = glob.glob(os.path.join(args.data_path, '**', 'papyrus', '*.jpg'), recursive=True)
img_map = {}
for image in images:
    img_id = image.split(os.sep)[-3]
    img_map.setdefault(img_id, []).append(image)

os.makedirs(args.output_path, exist_ok=True)
for img_id, images in tqdm(img_map.items()):
    im_info = re.search(r'(.+)_([rv])_([IRCL]+)', img_id)
    dirname = os.path.join(args.output_path, im_info.group(3) + im_info.group(2).upper())
    os.makedirs(dirname, exist_ok=True)
    for idx in range(len(images)):
        with Image.open(images[idx]) as f:
            img = white_to_transparency(f)
            new_img_path = os.path.join(args.output_path, dirname, f'{im_info.group(1)}.png')
            if len(images) > 1:
                new_img_path = os.path.join(args.output_path, dirname, f'{im_info.group(1)}-{idx}.png')

            img.save(new_img_path, optimize=True)
