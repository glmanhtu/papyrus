import glob
import logging
import os
import re

from torch.utils.data import Dataset

from exception.data_exception import PatchNotExtractableException
from utils import data_utils
from utils.data_utils import read_image, add_items_to_group

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

# exclude_imgs = ['2969c_IRV', '2969c_IRR', '2934c_IRR', '1306b_IRR', '1353f_IRV', '1353f_IRR', '0567o_IRR',
#                 '1250c_IRR', '1250c_IRV', '1224n_IRV', '1224n_IRR', '1317d_IRV', '1317d_IRR', '2841b_IRR',
#                 '2841b_IRV', '1445_IRR', '1445_COLR', '1445_IRV', '1445_COLV', '1278c_IRR', '2888e_COLV',
#                 '2888e_IRV', '0808b_IRV', '0808b_IRR', '0808e_IRV', '0808e_IRR', '1228a_IRR', '1228a_IRV',
#                 '2911b_IRR', '2911b_IRV', '0808c_IRV', '0808c_IRR', '2969e_IRR', '2969e_IRV', '2915_IRV',
#                 '2915_IRR', '1322m_IRR', '2996c_IRR', '2996c_IRV', '1290k_IRR', '1290k_IRV', '2930b_IRR',
#                 '2930b_IRV', '2916c_IRV', '2916c_IRR', '2949c_IRV', '2949c_IRR', '1438o_IRV', '0808a_IRV',
#                 '0808a_IRR', '2913_IRR', '2913_IRV', '2881d_IRV', '2881d_IRR', '2916b_IRV', '2916b_IRR',
#                 '1228d_IRV', '1228d_IRR', '1316c_IRR', '1316c_IRV', '1322o_IRV', '1322o_IRR', '2974_IRR',
#                 '2974_IRV', '2733e_IRR', '2733e_IRV', '2881c_COLR', '2881c_IRR', '2881c_IRV', '2916a_IRV',
#                 '2916a_IRR', '1306h_IRR', '2855e_COLV', '2855e_IRR', '2855e_COLR', '2838g_COLV', '2949a_IRR',
#                 '2867f_IRV', '2867f_COLR', '2867f_COLV', '1322k_IRV', '1322k_IRR', '2882b_COLR', '2882b_COLV',
#                 '2882b_IRR', '0567s_IRV', '2881b_IRV', '0794_IRV', '0794_IRR', '2969b_IRV', '2733r_IRR', '2733r_IRV',
#                 '2735a_IRV', '2735a_IRR', '1220d_1225e_IRR', '2850b_IRR', '2850b_IRV', '0567r_IRV', '0271p_IRV',
#                 '0271p_IRR', '2926_IRR', '2926_IRV', '1382i_IRR', '1382i_IRV', '0567o_0567p_0567q_IRV',
#                 '0567o_0567p_0567q_IRR', '2849d_IRR', '2849d_IRV', '1322l_IRV', '1322l_COLV', '2969d_IRR',
#                 '2969d_IRV', '1207_1209b_IRV', '2970a_IRR', '2970a_IRV', '2853c_IRR', '2975b_IRR', '2975b_IRV',
#                 '1223d_IRV', '1290u_IRV', '0779_1257_IRV', '0243_IRV', '0243_IRR', '2855b_COLR',
#                 '1317c_1321f_1321k_IRR', '2735i_IRR', '2735i_IRV', '1444a_1444b_IRR', '1444a_1444b_IRV',
#                 '2932_IRR', '2932_IRV', '2952b_IRV', '1224i_IRV', '1224i_IRR', '2949b_IRR', '2882c_COLV',
#                 '2882c_IRR', '1290r_IRV', '1290r_IRR', '0567n_IRV', '2859a_COLV', '2859a_IRV', '2859a_COLR',
#                 '2950b_IRV', '1438g_IRR', '1316d_IRV', '1316d_IRR', '2735j_IRR', '2735j_IRV', '1290w_IRV',
#                 '0811c_1442_IRR', '1223e_1224d_IRV', '1223e_1224d_IRR', '3007_IRR', '3007_IRV', '2970b_IRV',
#                 '2970b_IRR', '2875f_IRV', '2859b_COLV', '2859b_IRV', '2859b_COLR', '1378i_IRV',
#                 '0567s_IRR', '0567s_COLR', '0567s_COLV']
exclude_imgs = ['0567s_COLR']


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
        add_items_to_group(fragment_ids, groups)

    return groups


class InfraredDataset(Dataset):
    def __init__(self, dataset_path: str, transforms, patch_size=224, proportion=(0, 1),
                 patch_bg_threshold=0.6, file_type_filter='COLR'):
        self.dataset_path = dataset_path
        assert os.path.isdir(self.dataset_path)

        self.groups = extract_relations(dataset_path)
        self.fragment_ids = {}
        for idx, group in enumerate(self.groups):
            for fragment_id in group:
                self.fragment_ids[fragment_id] = idx

        image_pattern = os.path.join(dataset_path, '**', '*.jpg')
        files = glob.glob(image_pattern, recursive=True)

        papyri = {}
        for file in files:
            file_name = os.path.splitext(os.path.basename(file))[0]
            if file_type_filter not in file_name:
                continue

            if file_name in exclude_imgs:
                continue

            papyrus_id = self.get_papyrus_id(file_name)
            papyri.setdefault(papyrus_id, []).append(file)

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
