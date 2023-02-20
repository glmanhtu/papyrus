import numpy as np
import random


def extract_random_patch(image, patch_size):
    # Get the dimensions of the image
    height, width, channels = image.shape

    # Calculate the maximum x and y coordinates to ensure the patch fits inside the image
    max_x = width - patch_size
    max_y = height - patch_size

    if max_y < 0 or max_x < 0:
        raise Exception(f"Image {image.shape} too small!")

    # Get a random x and y coordinate within the maximum values
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Extract the patch from the image
    patch = image[y:y+patch_size, x:x+patch_size]

    return patch


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]