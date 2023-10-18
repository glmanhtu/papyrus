import argparse
import glob
import os
import re

import imagesize
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_dir', type=str, help='Path to the original dataset', required=True)
parser.add_argument('--ruler_length', type=int, default=700)
parser.add_argument('--output_dir', type=str, help='Path to the output folder', required=True)

args = parser.parse_args()

fragments = {}
for fragment_name in os.listdir(args.dataset_dir):
    folder_dir = os.path.join(args.dataset_dir, fragment_name)
    if not os.path.isdir(folder_dir):
        continue
    fragment_pattern = re.search(r'(.+)_([rv])_(IR|CL).*', fragment_name)
    if fragment_pattern is None:
        print(f'Cant read fragment: {folder_dir}')
    fragment_id = fragment_pattern.group(1)
    rv = fragment_pattern.group(2)
    ir_cl = fragment_pattern.group(3)

    ruler_imgs = glob.glob(os.path.join(folder_dir, '*ruler*', '*.jpg'))
    if len(ruler_imgs) > 0:
        assert len(ruler_imgs) == 1

        width, height = imagesize.get(ruler_imgs[0])
        ruler_lengths = max(width, height)
    else:
        ruler_lengths = 0

    if os.path.exists(os.path.join(folder_dir, 'cm papyrus')):
        os.replace(os.path.join(folder_dir, 'cm papyrus'), os.path.join(folder_dir, 'papyrus'))
    papyrus_images = glob.glob(os.path.join(folder_dir, 'papyrus', '*.jpg'))
    assert len(papyrus_images) > 0, f'Papyrus not found in {folder_dir}'

    record = {'RV': rv, 'IRCL': ir_cl, 'ruler_length': ruler_lengths, 'images': papyrus_images}
    fragments.setdefault(fragment_id, []).append(record)

for fragment_id in fragments:
    ruler_lengths = [x['ruler_length'] for x in fragments[fragment_id] if x['ruler_length'] > 0]
    ruler_length = sum(ruler_lengths) / len(ruler_lengths)
    for record in fragments[fragment_id]:
        record_ruler_length = ruler_length if record['ruler_length'] == 0 else record['ruler_length']
        scale = args.ruler_length / record_ruler_length
        for idx, img_path in enumerate(record['images']):
            with Image.open(img_path) as f:
                img = f.convert('RGB')

            new_w, new_h = int(img.width * scale), int(img.height * scale)
            img = img.resize((new_w, new_h))
            new_img_path = os.path.join(args.output_dir, fragment_id, f'{record["RV"]}_{record["IRCL"]}_{idx}.jpg')
            os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
            img.save(new_img_path)
