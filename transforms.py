import numpy as np
import torchvision

from utils import UnableToCrop


def compute_white_percentage(img, ref_size=224):
    gray = img.convert('L')
    if gray.width > ref_size:
        gray = gray.resize((ref_size, ref_size))
    gray = np.asarray(gray)
    white_pixel_count = np.sum(gray > 250)
    total_pixels = gray.shape[0] * gray.shape[1]
    return white_pixel_count / total_pixels


class CustomRandomCrop:
    def __init__(self, crop_size, white_percentage_limit=0.6, max_retry=1000, im_path=''):
        self.cropper = torchvision.transforms.RandomCrop(crop_size, pad_if_needed=True, fill=255)
        self.white_percentage_limit = white_percentage_limit
        self.max_retry = max_retry
        self.im_path = im_path

    def crop(self, img):
        current_retry = 0
        curr_w_p = 0
        while current_retry < self.max_retry:
            out = self.cropper(img)
            curr_w_p = compute_white_percentage(out)
            if curr_w_p <= self.white_percentage_limit:
                return out
            current_retry += 1
        raise UnableToCrop(f'Unable to crop, curr wp: {curr_w_p}', im_path=self.im_path)

    def __call__(self, img):
        return self.crop(img)
