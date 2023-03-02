import glob
import itertools
import logging
import os
import random
import re

from torch.utils.data import Dataset

from exception.data_exception import PatchNotExtractableException
from utils import data_utils
from utils.data_utils import read_image, minmax_split_chunks

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

excludes = ['4458br_22', '1253_19r_23', '1368R_11', '1368R_27', '2153dr_5', '2228r_32', '4800br_7', '4800br_8',
            '102r_21', '102r_23', '102r_26', '102r_28', '102r_25', '102r_29', '102r_27', '4857ar_47',
            '7219r_6', '7205r_5', '7205r_9']


def get_papyrus_id(fragment):
    papyrus_id = fragment.split('_')[0]

    tmp = re.search('[A-z]', papyrus_id)

    if tmp is not None:
        index_first_character = re.search('[A-z]', papyrus_id).start()
        papyrus_id = papyrus_id[:index_first_character]

    return papyrus_id


class MichiganDataset(Dataset):

    def __init__(self, dataset_path: str, transforms, patch_size=224, proportion=(0, 0.8),
                 only_recto=True, min_fragments_per_papyrus=2, patch_bg_threshold=0.5):
        self.dataset_path = dataset_path
        assert os.path.isdir(self.dataset_path)
        image_pattern = os.path.join(dataset_path, '**', '*.png')
        files = glob.glob(image_pattern, recursive=True)

        papyri = {}
        for file in files:
            file_name = os.path.splitext(os.path.basename(file))[0]
            if only_recto and 'r' not in file_name.lower():
                continue
            if file_name in excludes:
                continue
            papyrus_id = get_papyrus_id(file_name)
            papyri.setdefault(papyrus_id, []).append(file)

        for k, v in list(papyri.items()):
            if len(v) < min_fragments_per_papyrus:
                del papyri[k]
            if k == '':
                del papyri[k]

        papyrus_ids = list(sorted(papyri.keys()))
        p_from, p_to = proportion
        d_size = len(papyrus_ids)
        self.ids = papyrus_ids[int(d_size * p_from):int(d_size * p_to)]

        # Re-balance fragments
        for k, v in list(papyri.items()):
            papyri[k] = list(minmax_split_chunks(papyri[k]))

        self.patch_size = patch_size

        data = []
        for papyrus_id in self.ids:
            for anchor in papyri[papyrus_id]:
                positive_list = papyri[papyrus_id]
                negative_list = [papyri[x] for x in self.ids if x != papyrus_id]
                negative_list = list(itertools.chain.from_iterable(negative_list))
                data.append((positive_list, anchor, negative_list))

        self.data = data
        self.patch_bg_threshold = patch_bg_threshold
        self.transforms = transforms
        self.get_papyrus_id = get_papyrus_id

    def __len__(self):
        return len(self.data)

    def get_patch_by_id(self, img_id):
        img_path = os.path.join(self.dataset_path, f"{img_id}.png")
        return self.get_patch([img_path])[0]

    def get_patch(self, img_list):
        image_path = ''
        while len(img_list) > 0:
            image_path = random.choice(img_list)
            try:
                img = read_image(image_path)
                return data_utils.extract_random_patch(img, self.patch_size, self.patch_bg_threshold), image_path
            except PatchNotExtractableException:
                logging.error(f"Could not extract patch from image {image_path}, retry another image...")
                img_list.remove(image_path)
        raise Exception('Could not extract any patch. Last img: ' + image_path)

    def __getitem__(self, idx):
        positive_list, anchor, negative_list = self.data[idx]
        positive_list = list(itertools.chain.from_iterable(positive_list))
        negative_list = list(itertools.chain.from_iterable(negative_list))

        positive_patch, pos_img_path = self.get_patch(positive_list)
        positive_image = os.path.splitext(os.path.basename(pos_img_path))[0]

        anchor_patch, anc_img_path = self.get_patch(anchor)
        anchor_image = os.path.splitext(os.path.basename(anc_img_path))[0]

        negative_patch, neg_img_path = self.get_patch(negative_list)
        negative_image = os.path.splitext(os.path.basename(neg_img_path))[0]

        return {
            "positive": self.transforms(positive_patch),
            "pos_image":  positive_image,

            "anchor": self.transforms(anchor_patch),
            "anc_image": anchor_image,

            "negative": self.transforms(negative_patch),
            "neg_image": negative_image,
        }
