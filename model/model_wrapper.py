import os

import numpy as np
import torch
from torch import nn

from criterions.optim import Optimizer, Scheduler
from utils.misc import map_location


class ModelWrapper:
    def __init__(self, args, model, is_train, device):
        self._name = args.name
        self._model = model
        self._args = args
        self._is_train = is_train
        self._save_dir = os.path.join(self._args.checkpoints_dir, self.name)

        if self._is_train:
            self._model.train()
        else:
            self._model.eval()

        self._init_train_vars()
        self._device = device

    @property
    def name(self):
        return self._name

    @property
    def is_train(self):
        return self._is_train

    def _init_train_vars(self):
        self._optimizer = Optimizer().get(self._model, self._args.optimizer, lr=self._args.lr,
                                          wd=self._args.weight_decay)
        self.lr_scheduler = Scheduler().get(self._args.lr_policy, self._optimizer, step_size=self._args.lr_decay_epochs)

    def load(self):
        # load feature extractor
        self._load_network(self._model, self.name)
        self._load_optimizer(self._optimizer, self.name)

    def existing(self):
        return self._check_model(self.name)

    def get_current_lr(self):
        lr = []
        for param_group in self._optimizer.param_groups:
            lr.append(param_group['lr'])
        print('current learning rate: {}'.format(np.unique(lr)))

    def save(self):
        """
        save network, the filename is specified with the sofar tasks and iteration
        """
        self._save_network(self._model, self.name)
        # save optimizers
        self._save_optimizer(self._optimizer, self.name)

    def _save_optimizer(self, optimizer, optimizer_label):
        save_filename = 'opt_%s.pth' % optimizer_label
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def _load_optimizer(self, optimizer, optimizer_label):
        load_filename = 'opt_%s.pth' % optimizer_label
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(load_path), 'Weights file %s not found!' % load_path

        optimizer.load_state_dict(torch.load(load_path))
        print('loaded optimizer: %s' % load_path)

    def _save_network(self, network, network_label):
        save_filename = 'net_%s.pth' % network_label
        save_path = os.path.join(self._save_dir, save_filename)
        save_dict = network.state_dict()
        torch.save(save_dict, save_path)
        print('saved net: %s' % save_path)

    def load_network(self, pretrained_checkpoint):
        assert os.path.exists(pretrained_checkpoint), 'Weights file %s not found ' % pretrained_checkpoint
        checkpoint = torch.load(pretrained_checkpoint, map_location=map_location(self._args.cuda))
        self._model.load_state_dict(checkpoint)
        print('loaded net: %s' % pretrained_checkpoint)

    def _check_model(self, network_label):
        load_filename = 'net_%s.pth' % network_label
        load_path = os.path.join(self._save_dir, load_filename)
        return os.path.exists(load_path)

    def _load_network(self, network, network_label):
        load_filename = 'net_%s.pth' % network_label
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(load_path), 'Weights file %s not found ' % load_path
        checkpoint = torch.load(load_path, map_location=map_location(self._args.cuda))
        network.load_state_dict(checkpoint)
        print('loaded net: %s' % load_path)

    def set_train(self):
        self._model.train()
        self._is_train = True

    def set_eval(self):
        self._model.eval()
        self._is_train = False

    def compute_loss(self, batch_data):
        positive_images = batch_data['positive'].to(self._device, non_blocking=True)
        anchor_images = batch_data['anchor'].to(self._device, non_blocking=True)
        negative_images = batch_data['negative'].to(self._device, non_blocking=True)

        criteria = nn.TripletMarginLoss(margin=0.2)
        with torch.set_grad_enabled(self._is_train):
            pos_features = self._model(positive_images)
            anc_features = self._model(anchor_images)
            neg_features = self._model(negative_images)

            loss = criteria(anc_features, pos_features, neg_features)
            return loss, (pos_features, anc_features, neg_features)

    def optimise_params(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()