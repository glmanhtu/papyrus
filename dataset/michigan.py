import glob
import logging
import os
import re

from torch.utils.data import Dataset

from exception.data_exception import PatchNotExtractableException
from utils import data_utils
from utils.data_utils import read_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

excludes = ['4458br_22', '102r_25', '102r_29', '102r_27']


def get_papyrus_id(fragment):
    papyrus_id = fragment.split('_')[0]

    tmp = re.search('[A-z]', papyrus_id)

    if tmp is not None:
        index_first_character = re.search('[A-z]', papyrus_id).start()
        papyrus_id = papyrus_id[:index_first_character]

    return papyrus_id


class MichiganDataset(Dataset):

    def __init__(self, dataset_path: str, transforms, patch_size=224, proportion=(0, 0.8), only_recto=True):
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
            if papyrus_id not in papyri:
                papyri[papyrus_id] = []
            papyri[papyrus_id].append(file)

        for k, v in list(papyri.items()):
            if k == '':
                del papyri[k]

        papyrus_ids = list(sorted(papyri.keys()))
        p_from, p_to = proportion
        d_size = len(papyrus_ids)
        self.ids = papyrus_ids[int(d_size * p_from):int(d_size * p_to)]
        self.patch_size = patch_size

        data = []
        for papyrus_id in self.ids:
            for img in papyri[papyrus_id]:
                data.append(img)

        self.data = data
        self.transforms = transforms
        self.get_papyrus_id = get_papyrus_id
        self.bad_imgs = []

    def __len__(self):
        return len(self.data)

    def get_patch_by_id(self, img_id):
        img_path = os.path.join(self.dataset_path, f"{img_id}.png")
        return self.get_patch(img_path)

    def get_patch(self, image_path):
        img = read_image(image_path)
        return data_utils.extract_random_patch(img, self.patch_size)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        positive_image = os.path.splitext(os.path.basename(img_path))[0]
        try:
            positive_patch = self.get_patch(img_path)

            anchor_patch = self.get_patch(img_path)

            return {
                "positive": self.transforms(positive_patch),
                "pos_image":  positive_image,

                "anchor": self.transforms(anchor_patch),
                "anc_image": positive_image,
            }
        except PatchNotExtractableException:
            self.bad_imgs.append(positive_image)
