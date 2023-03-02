import glob
import logging
import os
import random
import itertools
import re

from torch.utils.data import Dataset

from exception.data_exception import PatchNotExtractableException
from utils import data_utils
from utils.data_utils import read_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

excludes = ['2855e', '2855b', '2882b', '2881c']     # Incorrect segmentation samples
excludes += ['0567s']   # Too small


def get_fragment_id(file_name):
    id_search = re.search(r'(\d+)', file_name)
    if id_search:
        return id_search.group(1)
    raise Exception('Pattern incorrect ' + file_name)


def extract_relations(dataset_path):
    """
    There are some fragments that the papyrologists have put together by hand in the database. These fragments
    are named using the pattern of <fragment 1>_<fragment 2>_<fragment 3>...
    Since they belong to the same papyrus, we should put them to the same category
    @param dataset_path:
    """

    groups = []

    for dir_name in sorted(os.listdir(dataset_path)):
        name_components = dir_name.split("_")
        fragment_ids = [get_fragment_id(x) for x in name_components]
        reference_group = None
        for group in groups:
            for fragment_id in fragment_ids:
                if fragment_id in group:
                    reference_group = group
                    break
            if reference_group is not None:
                break
        if reference_group is not None:
            for fragment_id in fragment_ids:
                reference_group.add(fragment_id)
        else:
            groups.append(set(fragment_ids))

    return groups


class InfraredDataset(Dataset):
    def __init__(self, dataset_path: str, transforms, patch_size=224, proportion=(0, 1),
                 only_recto=True, patch_bg_threshold=0.5):
        self.dataset_path = dataset_path
        assert os.path.isdir(self.dataset_path)

        self.groups = extract_relations(dataset_path)
        self.fragment_ids = {}
        for idx, group in enumerate(self.groups):
            for fragment_id in group:
                self.fragment_ids[fragment_id] = idx

        image_pattern = os.path.join(dataset_path, '**', '*.png')
        files = glob.glob(image_pattern, recursive=True)

        papyri = {}
        for file in files:
            file_name = os.path.splitext(os.path.basename(file))[0]
            if only_recto and 'COLR' not in file_name:
                continue

            if file_name.rsplit("_")[0] in excludes:
                continue

            papyrus_id = self.get_papyrus_id(file_name)
            if papyrus_id not in papyri:
                papyri[papyrus_id] = []
            papyri[papyrus_id].append(file)

        papyrus_ids = list(sorted(papyri.keys()))
        p_from, p_to = proportion
        d_size = len(papyrus_ids)
        self.patch_bg_threshold = patch_bg_threshold
        self.ids = papyrus_ids[int(d_size * p_from):int(d_size * p_to)]
        self.patch_size = patch_size

        data = []
        for papyrus_id in self.ids:
            for img in papyri[papyrus_id]:
                data.append(img)

        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def get_papyrus_id(self, file_name):
        fragment_name = file_name.rsplit('_', 1)[0]     # Remove _COLR, _COLV, etc
        fragment_ids = [get_fragment_id(x) for x in fragment_name.split('_')]
        return self.fragment_ids[fragment_ids[0]]

    def get_patch_by_id(self, img_id):
        fragment_name = img_id.rsplit('_', 1)[0]     # Remove _COLR, _COLV, etc
        img_path = os.path.join(self.dataset_path, fragment_name, f"{img_id}.png")
        return self.get_patch(img_path)

    def get_patch(self, image_path):
        img = read_image(image_path)
        try:
            return data_utils.extract_random_patch(img, self.patch_size, background_threshold=self.patch_bg_threshold)
        except PatchNotExtractableException:
            raise Exception(f'Unable to extract patch from image {image_path}')

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        positive_patch = self.get_patch(img_path)

        anchor_patch = self.get_patch(img_path)

        return {
            "positive": self.transforms(positive_patch),
            "pos_image": img_id,

            "anchor": self.transforms(anchor_patch),
            "anc_image": img_id,
        }
