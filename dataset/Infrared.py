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
from utils.misc import get_papyrus_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


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

    relation_map = {}

    for dir_name in os.listdir(dataset_path):
        name_components = dir_name.split("_")
        for name in name_components:
            fragment_id = get_fragment_id(name)
            relation_map.setdefault(fragment_id, set([])).add(name)
    return relation_map


class InfraredDataset(Dataset):
    def __init__(self, dataset_path: str, transforms, patch_size=224, proportion=(0, 0.8), only_recto=True):
        self.dataset_path = dataset_path
        assert os.path.isdir(self.dataset_path)

        relation_map = extract_relations(dataset_path)

        image_pattern = os.path.join(dataset_path, '**', '*.png')
        files = glob.glob(image_pattern, recursive=True)

        papyri = {}
        for file in files:
            file_name = os.path.splitext(os.path.basename(file))[0]
            if only_recto and 'COLR' not in file_name.lower():
                continue
            papyrus_id = get_papyrus_id(file_name)
            if papyrus_id not in papyri:
                papyri[papyrus_id] = []
            papyri[papyrus_id].append(file)

        papyrus_ids = list(sorted(papyri.keys()))
        p_from, p_to = proportion
        d_size = len(papyrus_ids)
        self.ids = papyrus_ids[int(d_size * p_from):int(d_size * p_to)]

        # Re-balance fragments
        for k, v in list(papyri.items()):
            papyri[k] = list(split_chunks(papyri[k]))

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

    def get_patch_by_id(self, img_id):
        img_path = os.path.join(self.dataset_path, f"{img_id}.png")
        return self.get_patch([img_path])[0]

    def get_patch(self, img_list):
        image_path = ''
        while len(img_list) > 0:
            image_path = random.choice(img_list)
            try:
                img = read_image(image_path)
                return data_utils.extract_random_patch(img, self.patch_size), image_path
            except PatchNotExtractableException:
                # logging.error(f"Could not extract patch from image {image_path}, retry another image...")
                img_list.remove(image_path)
        raise Exception('Could not extract any patch. Last img: ' + image_path)

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
