import glob
import os
import re
from enum import Enum
from typing import Callable, Optional, Union

import imagesize
from PIL import Image
from ml_engine.data.grouping import add_items_to_group
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"

    def is_train(self):
        return self.value == 'train'

    def is_val(self):
        return self.value == 'validation'

    def is_test(self):
        return self.value == 'test'

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


def parse_name(name: str, is_train: bool):
    groups = re.search(r'^([\w\']+)_([rv])_(\w+)(\s.+)?$', name)
    if groups:
        fragment, rv, col = groups.group(1), groups.group(2), groups.group(3)
        if is_train:
            groups = re.search(r'^(\d+)?(\w+)?', fragment)
            if groups.group(1):
                fragment = groups.group(1)
            else:
                fragment = groups.group(2)
        return fragment, rv, col
    raise ValueError(f"Fragment name {name} not recognized")


class MergeDataset(Dataset):
    def __init__(self, datasets, transform):
        self.data = []
        self.data_labels = []

        for dataset in datasets:
            self.data.extend(dataset.data)
            self.data_labels.extend(dataset.data_labels)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fragment = self.data[idx]

        with Image.open(fragment) as img:
            image = self.transform(img.convert('RGB'))

        label = self.data_labels[idx]
        return image, label


class GeshaemPatch(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        root: str,
        split: "GeshaemPatch.Split",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        include_verso=False,
        min_size_limit=112,
        base_idx=0
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.root_dir = root

        self.fragment_to_group = {}
        self.fragment_to_group_id = {}

        fragments, groups = self.load_dataset(include_verso, min_size_limit, split)

        for idx, group in enumerate(groups):
            if len(group) < 2 and split.is_val():
                # We only evaluate the fragments that we know they are belongs to a certain groups
                # If the group have only one element, which means that very likely that we don't know
                # which group this element belongs to, so we skip it
                continue
            for fragment in group:
                self.fragment_to_group_id[fragment] = idx
                for fragment2 in group:
                    self.fragment_to_group.setdefault(fragment, set([])).add(fragment2)

        self.fragments = sorted(fragments.keys())
        self.fragment_idx = {x: i for i, x in enumerate(self.fragments)}

        self.data = []
        self.data_labels = []
        for idx, fragment in enumerate(self.fragments):
            data, labels = [], []
            for img_path in sorted(fragments[fragment]):
                image_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))
                fragment, rv, col = parse_name(image_name, split.is_train())
                if fragment not in self.fragment_to_group:
                    continue

                labels.append(idx + base_idx)
                data.append(img_path)

            self.data.extend(data)
            self.data_labels.extend(labels)

    def get_group_id(self, fragment_id: int) -> int:
        fragment = self.fragments[fragment_id]
        return self.fragment_to_group_id[fragment]

    def remove_duplicate(self, fragment_ids, global_keys):
        keys = {}
        for fragment in fragment_ids:
            groups = re.search(r'^(\d+)?(\w+)?', fragment)
            if groups.group(1):
                key = groups.group(1)
            else:
                key = groups.group(2)
            if key in global_keys:
                global_keys[key].append(fragment)
                continue
            keys.setdefault(key, []).append(fragment)
        global_keys.update(keys)
        return [keys[x][0] for x in keys]

    def load_dataset(self, include_verso, min_size_limit, split: _Split):
        fragments = {}
        groups = []
        keys = {}
        for img_path in sorted(glob.glob(os.path.join(self.root_dir, '**', '*.jpg'), recursive=True)):
            if img_path.split(os.sep)[-3] != 'papyrus':
                continue
            image_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))
            fragment, rv, col = parse_name(image_name, split.is_train())
            if rv.upper() == 'V' and not include_verso:
                continue

            fragment_ids = fragment.split("_")
            if split.is_val():
                # Exclude the pairs that has been trained in training mode
                fragment_ids = self.remove_duplicate(fragment_ids, keys)
            add_items_to_group(fragment_ids + [fragment], groups)
            if split.is_train() and len(fragment_ids) > 1:
                # We exclude the assembled fragments in training to prevent data leaking
                continue

            width, height = imagesize.get(img_path)
            if width * height < min_size_limit * min_size_limit:
                continue

            fragments.setdefault(fragment, []).append(img_path)

        return fragments, groups

    @property
    def split(self) -> "GeshaemPatch.Split":
        return self._split

    def __getitem__(self, index: int):
        img_path = self.data[index]

        with Image.open(img_path) as f:
            image = f.convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, self.data_labels[index]

    def __len__(self) -> int:
        return len(self.data)

