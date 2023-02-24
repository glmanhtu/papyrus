import glob
import logging
import os
import random
import itertools

import cv2
from torch.utils.data import Dataset

from exception.data_exception import PatchNotExtractableException
from utils import data_utils
from utils.misc import get_papyrus_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


def read_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class MichiganDataset(Dataset):

    def __init__(self, dataset_path: str, transforms, patch_size=224, proportion=(0, 0.8),
                 only_recto=True, min_fragments_per_papyrus=2):
        self.dataset_path = dataset_path
        assert os.path.isdir(self.dataset_path)
        image_pattern = os.path.join(dataset_path, '**', '*.png')
        files = glob.glob(image_pattern, recursive=True)

        papyri = {}
        for file in files:
            file_name = os.path.splitext(os.path.basename(file))[0]
            if only_recto and 'r' not in file_name.lower():
                continue
            papyrus_id = get_papyrus_id(file_name)
            if papyrus_id not in papyri:
                papyri[papyrus_id] = []
            papyri[papyrus_id].append(file)

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
            papyri[k] = list(data_utils.chunks(papyri[k], min_fragments_per_papyrus))

        self.patch_size = patch_size

        data = []
        for papyrus_id in self.ids:
            for anchor in papyri[papyrus_id]:
                positive_list = papyri[papyrus_id]
                negative_list = [papyri[x] for x in self.ids if x != papyrus_id]
                negative_list = list(itertools.chain.from_iterable(negative_list))
                data.append((positive_list, anchor, negative_list))

        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def get_patch(self, img_list):
        while len(img_list) > 0:
            image_path = random.choice(img_list)
            try:
                img = read_image(image_path)
                return data_utils.extract_random_patch(img, self.patch_size), image_path
            except PatchNotExtractableException:
                logging.error(f"Could not extract patch from image {image_path}, retry another image...")
                img_list.remove(image_path)
        raise Exception('Could not extract any patch...')

    def __getitem__(self, idx):
        positive_list, anchor, negative_list = self.data[idx]
        positive_list = list(itertools.chain.from_iterable(positive_list))
        negative_list = list(itertools.chain.from_iterable(negative_list))

        positive_patch, pos_img_path = self.get_patch(positive_list)
        positive_image = os.path.splitext(os.path.basename(pos_img_path))[0]
        positive_papyrus_id = get_papyrus_id(positive_image)

        anchor_patch, anc_img_path = self.get_patch(anchor)
        anchor_image = os.path.splitext(os.path.basename(anc_img_path))[0]
        anchor_papyrus_id = get_papyrus_id(anchor_image)

        negative_patch, neg_img_path = self.get_patch(negative_list)
        negative_image = os.path.splitext(os.path.basename(neg_img_path))[0]
        negative_papyrus_id = get_papyrus_id(negative_image)

        return {
            "positive": self.transforms(positive_patch),
            "pos_image":  positive_image,
            "pos_fragment": positive_papyrus_id,

            "anchor": self.transforms(anchor_patch),
            "anc_image": anchor_image,
            "anc_fragment": anchor_papyrus_id,

            "negative": self.transforms(negative_patch),
            "neg_image": negative_image,
            "neg_fragment": negative_papyrus_id
        }
