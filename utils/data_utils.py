import cv2
import numpy as np
import random

from exception.data_exception import PatchNotExtractableException


def padding_image(img, new_size, color=(0, 0, 0)):
    old_image_height, old_image_width, channels = img.shape
    new_image_width, new_image_height = new_size
    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
    return result


def extract_random_patch(image, patch_size, background_threshold=0.6, max_retries=100, current_retries=0):
    # Get the dimensions of the image
    height, width, channels = image.shape

    # Calculate the maximum x and y coordinates to ensure the patch fits inside the image
    max_x = width - patch_size
    max_y = height - patch_size

    if current_retries > max_retries:
        raise PatchNotExtractableException()

    if max_y < 0 or max_x < 0:
        new_width = int(width if width >= patch_size else patch_size * 1.5)
        new_height = int(height if height >= patch_size else patch_size * 1.5)
        new_image = padding_image(image, (new_width, new_height), color=(255, 255, 255))
        return extract_random_patch(new_image, patch_size, background_threshold, max_retries, current_retries)

    # Get a random x and y coordinate within the maximum values
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Extract the patch from the image
    patch = image[y:y+patch_size, x:x+patch_size]

    patch_gray = cv2.bitwise_not(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY))
    patch_background_percentage = np.sum(patch_gray < 30) / (patch_size * patch_size)

    if patch_background_percentage > background_threshold:
        return extract_random_patch(image, patch_size, background_threshold, max_retries, current_retries + 1)

    return patch


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


def read_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def minmax_split_chunks(data, n_chunks=(2, 4)):
    min_chunk, max_chunks = n_chunks
    middle = (min_chunk + max_chunks) // 2
    if len(data) > max_chunks * 2:
        return chunks(data, max_chunks)
    elif len(data) > middle * 2:
        return chunks(data, middle)
    else:
        return chunks(data, min_chunk)


def add_items_to_group(items, groups):
    reference_group = {}
    for g_id, group in enumerate(groups):
        for fragment_id in items:
            if fragment_id in group and g_id not in reference_group:
                reference_group[g_id] = group

    if len(reference_group) > 0:
        reference_ids = list(reference_group.keys())
        for fragment_id in items:
            reference_group[reference_ids[0]].add(fragment_id)
        for g_id in reference_ids[1:]:
            for fragment_id in reference_group[g_id]:
                reference_group[reference_ids[0]].add(fragment_id)
            del groups[g_id]
    else:
        groups.append(set(items))
