import random
import re
import time
from typing import Dict, List

import numpy as np
import torch.nn.functional as F

import torch
from torch import Tensor
import pandas as pd

from utils import wi19_evaluate


class EarlyStop:

    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.losses = []
        self.best_loss = 99999999

    def should_stop(self, loss):
        self.losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
        if len(self.losses) <= self.n_epochs:
            return False
        best_loss_pos = self.losses.index(self.best_loss)
        if len(self.losses) - best_loss_pos <= self.n_epochs:
            return False
        return True


def map_location(cuda):
    if torch.cuda.is_available() and cuda:
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'


def display_terminal(iter_start_time, i_epoch, i_train_batch, num_batches, train_dict):
    t = (time.time() - iter_start_time)
    current_time = time.strftime("%H:%M", time.localtime(time.time()))
    output = "Time {}\tBatch Time {:.2f}\t Epoch [{}]([{}/{}])\t".format(current_time, t,
                                                                         i_epoch, i_train_batch, num_batches)
    for key in train_dict:
        output += '{} {:.4f}\t'.format(key, train_dict[key])
    print(output)


def display_terminal_eval(iter_start_time, i_epoch, eval_dict):
    t = (time.time() - iter_start_time)
    output = "\nEval Time {:.2f}\t Epoch [{}] \t".format(t, i_epoch)
    for key in eval_dict:
        output += '{} {:.4f}\t'.format(key, eval_dict[key])
    print(output + "\n")


def compute_distance_matrix(data: Dict[str, List[Tensor]], n_times_testing=5):
    distance_map = {}
    fragments = list(data.keys())
    for i in range(len(fragments)):
        for j in range(i, len(fragments)):
            source, target = fragments[i], fragments[j]
            n_items = min(len(data[source]), len(data[target]))
            n_times = max((len(data[source]) + len(data[target])) // 2, n_times_testing)
            distances = []
            for _ in range(n_times):
                source_features = torch.stack(random.sample(data[source], n_items))
                target_features = torch.stack(random.sample(data[target], n_items))
                similarity = F.cosine_similarity(source_features, target_features, dim=0)
                similarity_percentage = (similarity + 1) / 2   # As output of cosine_similarity ranging between [-1, 1]
                distances.append(similarity_percentage.mean().item())

            mean_distance = sum(distances) / len(distances)
            distance_map.setdefault(source, {})[target] = mean_distance
            distance_map.setdefault(target, {})[source] = mean_distance

    matrix = pd.DataFrame.from_dict(distance_map, orient='index').sort_index()
    return matrix.reindex(sorted(matrix.columns), axis=1)


def get_papyrus_id(fragment):
    papyrus_id = fragment.split('_')[0]

    tmp = re.search('[A-z]', papyrus_id)

    if tmp is not None:
        index_first_character = re.search('[A-z]', papyrus_id).start()
        papyrus_id = papyrus_id[:index_first_character]

    return papyrus_id


def get_metrics(similarity_matrix):
    papyrus_ids = [get_papyrus_id(x) for x in similarity_matrix.index]
    papyrus_set_indexes = list(set(papyrus_ids))
    papyrus_ids = [papyrus_set_indexes.index(x) for x in papyrus_ids]
    precision_at, recall_at, sorted_retrievals = wi19_evaluate.get_precision_recall_matrices(
        similarity_matrix.to_numpy(), np.array(papyrus_ids), remove_self_column=False)

    non_singleton_idx = sorted_retrievals.sum(axis=1) > 0
    mAP = wi19_evaluate.compute_map(precision_at[non_singleton_idx, :], sorted_retrievals[non_singleton_idx, :])
    top_1 = sorted_retrievals[:, 0].sum() / len(sorted_retrievals)
    pr_a_k10 = compute_pr_a_k(sorted_retrievals, 10)
    pr_a_k100 = compute_pr_a_k(sorted_retrievals, 100)
    # roc = wi19_evaluate.compute_roc(sorted_retrievals)
    return mAP, top_1, pr_a_k10, pr_a_k100


def compute_pr_a_k(sorted_retrievals, k):
    pr_a_k = sorted_retrievals[:, :k].sum(axis=1) / np.minimum(sorted_retrievals.sum(axis=1), k)
    return pr_a_k.sum() / len(pr_a_k)



