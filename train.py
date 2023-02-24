import gc
import time

import torch
import wandb
from torch.utils.data import DataLoader

from dataset.michigan import MichiganDataset
from model.model_factory import ModelsFactory
from options.train_options import TrainOptions
from utils.misc import EarlyStop, display_terminal, compute_similarity_matrix, get_metrics, compute_pr_a_k, \
    display_terminal_eval
from utils.transform import get_transforms, val_transforms

args = TrainOptions().parse()


class Trainer:
    def __init__(self):
        # wandb.init(group=args.group,
        #            project=args.wb_project,
        #            entity=args.wb_entity)
        # wandb.run.name = args.name
        # wandb.run.save()
        # wandb.config.update(args)
        device = torch.device('cuda' if args.cuda else 'cpu')

        self._model = ModelsFactory.get_model(args, is_train=True, device=device, dropout=args.dropout)
        transforms = get_transforms(args)
        dataset_train = MichiganDataset(args.michigan_dir, transforms, patch_size=args.image_size, proportion=(0, 0.8),
                                        only_recto=True, min_fragments_per_papyrus=2)
        self.data_loader_train = DataLoader(dataset_train, shuffle=True, num_workers=args.n_threads_train,
                                            batch_size=args.batch_size, drop_last=True)
        transforms = val_transforms(args)
        dataset_val = MichiganDataset(args.michigan_dir, transforms, patch_size=args.image_size, proportion=(0.8, 1),
                                      only_recto=True, min_fragments_per_papyrus=2)

        self.data_loader_val = DataLoader(dataset_val, shuffle=False, num_workers=args.n_threads_test,
                                          batch_size=args.batch_size)

        self.early_stop = EarlyStop(args.early_stop)
        print("Training sets: {} images".format(len(dataset_train)))
        print("Validating sets: {} images".format(len(dataset_val)))

        self._current_step = 0
        if not self._model.existing():
            self._train()
        self._model.load()

    def _train(self):
        self._last_save_time = time.time()
        best_m_ap = 0.
        for i_epoch in range(1, args.nepochs + 1):
            epoch_start_time = time.time()
            self._model.get_current_lr()
            # train epoch
            self._train_epoch(i_epoch)
            if args.lr_policy == 'step':
                self._model.lr_scheduler.step()

            if not i_epoch % args.n_epochs_per_eval == 0:
                continue

            val_dict = self._validate(i_epoch, self.data_loader_val)
            gc.collect()

            current_m_ap = val_dict['val/m_ap']
            if current_m_ap > best_m_ap:
                print("mAP improved, from {:.4f} to {:.4f}".format(best_m_ap, current_m_ap))
                best_m_ap = current_m_ap
                for key in val_dict:
                    wandb.run.summary[f'best_model/{key}'] = val_dict[key]
                self._model.save()  # save best model

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, args.nepochs, time_epoch, time_epoch / 60, time_epoch / 3600))

            if self.early_stop.should_stop(1 - current_m_ap):
                print(f'Early stop at epoch {i_epoch}')
                break

    def _train_epoch(self, i_epoch):
        self._model.set_train()
        losses = []
        for i_train_batch, train_batch in enumerate(self.data_loader_train):
            iter_start_time = time.time()

            train_loss, _ = self._model.compute_loss(train_batch)
            self._model.optimise_params(train_loss)
            losses.append(train_loss.item())

            # update epoch info
            self._current_step += 1

            if self._current_step % args.save_freq_iter == 0:
                save_dict = {
                    'train/loss': sum(losses) / len(losses),
                }
                wandb.log(save_dict, step=self._current_step)
                display_terminal(iter_start_time, i_epoch, i_train_batch, len(self.data_loader_train), save_dict)

    @staticmethod
    def add_features(img_features, fragment_features, images, fragments, features):
        for image_name, fragment, features in zip(images, fragments, features):
            feature_cpu = features.cpu()
            if image_name not in img_features:
                img_features[image_name] = []
            img_features[image_name].append(feature_cpu)
            if fragment not in fragment_features:
                fragment_features[fragment] = []
            fragment_features[fragment].append(feature_cpu)

    def _validate(self, i_epoch, val_loader):
        val_start_time = time.time()
        # set model to eval
        self._model.set_eval()
        val_losses = []
        img_features, frag_features = {}, {}
        for _ in range(2):
            for i_train_batch, batch in enumerate(val_loader):
                val_loss, (pos_features, anc_features, neg_features) = self._model.compute_loss(batch)
                val_losses.append(val_loss)
                self.add_features(img_features, frag_features, batch['pos_image'], batch['pos_fragment'], pos_features)
                self.add_features(img_features, frag_features, batch['anc_image'], batch['anc_fragment'], anc_features)
                self.add_features(img_features, frag_features, batch['neg_image'], batch['neg_fragment'], neg_features)

        df = compute_similarity_matrix(frag_features)
        wandb.log({'val_fragment_level': wandb.plots.HeatMap(df.columns, df.index, df.to_numpy(), show_text=False)})

        df = compute_similarity_matrix(img_features)
        wandb.log({'val_img_level': wandb.plots.HeatMap(df.columns, df.index, df.to_numpy(), show_text=False)})

        m_ap, top1, pr_a_k10, pr_a_k100 = get_metrics(df)

        val_dict = {
            'val/loss': sum(val_losses) / len(val_losses),
            'val/m_ap': m_ap,
            'val/top_1': top1,
            'val/pr_a_k10': pr_a_k10,
            'val/pr_a_k100': pr_a_k100
        }
        wandb.log(val_dict, step=self._current_step)
        display_terminal_eval(val_start_time, i_epoch, val_dict)

        # set model back to train
        self._model.set_train()
        return val_dict


if __name__ == "__main__":
    trainer = Trainer()
