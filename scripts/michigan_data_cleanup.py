import argparse
import glob
import os

import tqdm

parser = argparse.ArgumentParser('Michigan data collection script', add_help=False)
parser.add_argument('--data-path', required=True, type=str, help='path to dataset')
args = parser.parse_args()

files = glob.iglob(os.path.join(args.data_path, '**', '*.jpg'), recursive=True)

image_map = {}
for file in tqdm.tqdm(files):
    file_name_components = file.split(os.sep)
    im_name, rv, sum_det, sub_name, im_type, _, _ = file_name_components[-7:]
    image_map.setdefault(im_name, {}).setdefault(sub_name, []).append(file)

sub_keys = {}
keys = list(image_map.keys())
for i in tqdm.tqdm(range(len(keys))):
    for sub_key in image_map[keys[i]]:
        sub_keys.setdefault(sub_key, {}).setdefault(keys[i], image_map[keys[i]][sub_key])
for key in sub_keys:
    if len(sub_keys[key]) > 1:
        folders = sorted(sub_keys[key])
        for i in range(1, len(folders)):
            for im_link in sub_keys[key][folders[i]]:
                os.unlink(im_link)
