import glob
import logging
import os
import random
import re
import itertools

import cv2
import torch
from torch.utils.data import Dataset

from exception.data_exception import PatchNotExtractableException
from utils import data_utils


logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


def get_papyrus_id(fragment):
    papyrus_id = fragment.split('_')[0]

    tmp = re.search('[A-z]', papyrus_id)

    if tmp is not None:
        index_first_character = re.search('[A-z]', papyrus_id).start()
        papyrus_id = papyrus_id[:index_first_character]

    return papyrus_id


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
                return data_utils.extract_random_patch(img, self.patch_size)
            except PatchNotExtractableException:
                logging.error(f"Could not extract patch from image {image_path}, retry another image...")
                img_list.remove(image_path)
        raise Exception('Could not extract any patch...')

    def __getitem__(self, idx):
        positive_list, anchor, negative_list = self.data[idx]
        positive_list = list(itertools.chain.from_iterable(positive_list))
        negative_list = list(itertools.chain.from_iterable(negative_list))

        positive_patch = self.get_patch(positive_list)
        anchor_patch = self.get_patch(anchor)
        negative_patch = self.get_patch(negative_list)

        return {
            "positive": self.transforms(torch.from_numpy(positive_patch)),
            "anchor": self.transforms(torch.from_numpy(anchor_patch)),
            "negative": self.transforms(torch.from_numpy(negative_patch))
        }



