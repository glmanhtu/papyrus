import random

import torch

from utils.misc import compute_similarity_matrix, get_metrics


def generate_random_features(size):
    length = random.randint(1, 10)
    result = []
    for _ in range(length):
        result.append(torch.rand((size,), dtype=torch.float32))
    return result


img_features = {
    '59ar_1': generate_random_features(128),
    '59br_1': generate_random_features(128),
    '48va_4': generate_random_features(128),
    '48va_1': generate_random_features(128),
    '57r_1': generate_random_features(128)
}

df = compute_similarity_matrix(img_features)
map = get_metrics(df)
