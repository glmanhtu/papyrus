import random
import time
from typing import Dict, List
import torch.nn.functional as F

import torch
from torch import Tensor
import pandas as pd


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


def compute_similarity_matrix(data: Dict[str, List[Tensor]], n_times_testing=5):
    similarity_map = {}
    fragments = list(data.keys())
    for i in range(len(fragments)):
        for j in range(i, len(fragments)):
            source, target = fragments[i], fragments[j]
            n_items = min(len(data[source]), len(data[target]))
            n_times = max((len(data[source]) + len(data[target])) // 2, n_times_testing)
            similarities = []
            for _ in range(n_times):
                source_features = torch.stack(random.sample(data[source], n_items))
                target_features = torch.stack(random.sample(data[target], n_items))
                similarity = F.cosine_similarity(source_features, target_features, dim=0)
                similarity_percentage = (similarity + 1) / 2   # As output of cosine_similarity ranging between [-1, 1]
                similarities.append(similarity_percentage)

            mean_similarity = sum(similarities) / len(similarities)
            similarity_map.setdefault(source, {})[target] = mean_similarity
            similarity_map.setdefault(target, {})[source] = mean_similarity

    return pd.DataFrame.from_dict(similarity_map, orient='index')


def compute_map(similarity_matrix):
    return 0


def compute_pr_a_k(similarity_matrix, k):
    return 0
