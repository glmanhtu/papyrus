import os.path
import os.path
import time

import torch
from torch.utils.data import DataLoader

import wandb
from dataset.infrared import InfraredDataset
from dataset.michigan import MichiganDataset
from model.model_factory import ModelsFactory
from options.finetune_options import FinetuneOptions
from options.train_options import TrainOptions
from utils import wb_utils
from utils.misc import EarlyStop, display_terminal, compute_similarity_matrix, get_metrics, display_terminal_eval, \
    random_query_results
from utils.transform import get_transforms, val_transforms
from utils.wb_utils import create_heatmap

args = FinetuneOptions().parse()


class Trainer:
    def __init__(self, dtype):
        print(f'Current data type: {data_type}')
        device = torch.device('cuda' if args.cuda else 'cpu')
        self._dtype = dtype
        self._working_dir = os.path.join(args.checkpoints_dir, args.name, dtype)
        self._model = ModelsFactory.get_model(args, self._working_dir, is_train=True, device=device,
                                              dropout=args.dropout)
        self._model.load_network(args.pretrained_model)
        transforms = get_transforms()
        dataset_train = InfraredDataset(args.infrared_dir, transforms, patch_size=args.image_size, proportion=(0, 1),
                                        file_type_filter=dtype)
        self.data_loader_train = DataLoader(dataset_train, shuffle=True, persistent_workers=True,
                                            num_workers=args.n_threads_train,
                                            batch_size=args.batch_size, drop_last=True, pin_memory=True)
        transforms = val_transforms(args)
        dataset_val = InfraredDataset(args.infrared_dir, transforms, patch_size=args.image_size, proportion=(0, 1),
                                      file_type_filter=dtype)

        self.data_loader_val = DataLoader(dataset_val, shuffle=False, num_workers=args.n_threads_test,
                                          batch_size=args.batch_size)

        self.early_stop = EarlyStop(args.early_stop)
        print("Training sets: {} images".format(len(dataset_train)))
        print("Validating sets: {} images".format(len(dataset_val)))

        self._current_step = 0

    def is_trained(self):
        return self._model.existing()

    def set_current_step(self, step):
        self._current_step = step

    def load_pretrained_model(self):
        self._model.load()

    def train(self):
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

            val_dict, similar_df = self._validate(i_epoch, self.data_loader_val, mode=f'val/{self._dtype}')

            current_m_ap = val_dict[f'val/{self._dtype}/m_ap']
            if current_m_ap > best_m_ap:
                print("mAP improved, from {:.4f} to {:.4f}".format(best_m_ap, current_m_ap))
                best_m_ap = current_m_ap
                for key in val_dict:
                    wandb.run.summary[f'best_model/{key}'] = val_dict[key]
                self._model.save()  # save best model
                similar_df.to_csv(os.path.join(self._working_dir, 'similarity_matrix.csv'), encoding='utf-8')

                query_results = random_query_results(similar_df, self.data_loader_val.dataset, n_queries=5, top_k=25)
                wandb.log({f'best_prediction/{self._dtype}': wb_utils.generate_query_table(query_results, top_k=25)},
                          step=self._current_step)

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, args.nepochs, time_epoch, time_epoch / 60, time_epoch / 3600))

            if self.early_stop.should_stop(1 - current_m_ap):
                print(f'Early stop at epoch {i_epoch}')
                break

    def final_eval(self):
        val_dict, similar_df = self._validate(0, self.data_loader_val, mode=f'val/{self._dtype}')
        similar_df.to_csv(os.path.join(self._working_dir, 'similarity_matrix.csv'), encoding='utf-8')

    def _train_epoch(self, i_epoch):
        self._model.set_train()
        losses = []
        for i_train_batch, train_batch in enumerate(self.data_loader_train):
            iter_start_time = time.time()

            train_loss, _ = self._model.compute_loss(train_batch)
            self._model.optimise_params(train_loss)
            losses.append(train_loss.item() + 1)  # negative cosine similarity has range [-1, 1]

            # update epoch info
            self._current_step += 1

            if self._current_step % args.save_freq_iter == 0:
                save_dict = {
                    'train/loss': sum(losses) / len(losses),
                }
                losses.clear()
                wandb.log(save_dict, step=self._current_step)
                display_terminal(iter_start_time, i_epoch, i_train_batch, len(self.data_loader_train), save_dict)

    @staticmethod
    def add_features(img_features, images, features):
        for image_name, features in zip(images, features):
            if image_name not in img_features:
                img_features[image_name] = []
            img_features[image_name].append(features)

    def _validate(self, i_epoch, val_loader, mode='val', n_time_validates=3):
        val_start_time = time.time()
        # set model to eval
        self._model.set_eval()
        val_losses = []
        img_features = {}
        for i in range(n_time_validates):
            for i_train_batch, batch in enumerate(val_loader):
                val_loss, (pos_features, anc_features) = self._model.compute_loss(batch)
                val_losses.append(val_loss.item() + 1)  # negative cosine similarity has range [-1, 1]
                self.add_features(img_features, batch['pos_image'], pos_features)
                self.add_features(img_features, batch['anc_image'], anc_features)
            print(f'Finished the evaluating {i + 1}/{n_time_validates}')

        img_features = {k: torch.stack(v) for k, v in img_features.items()}
        similar_df = compute_similarity_matrix(img_features)
        m_ap, top1, pr_a_k10, pr_a_k100 = get_metrics(similar_df, val_loader.dataset.get_papyrus_id)

        val_dict = {
            f'{mode}/loss': sum(val_losses) / len(val_losses),
            f'{mode}/m_ap': m_ap,
            f'{mode}/top_1': top1,
            f'{mode}/pr_a_k10': pr_a_k10,
            f'{mode}/pr_a_k100': pr_a_k100
        }
        wandb.log(val_dict, step=self._current_step)
        display_terminal_eval(val_start_time, i_epoch, val_dict)

        return val_dict, similar_df


if __name__ == "__main__":
    for data_type in ['COLR', 'COLV', 'IRR', 'IRV']:
        run = wandb.init(group=args.group,
                         name=f'{args.name}-{data_type}',
                         project=args.wb_project,
                         entity=args.wb_entity,
                         config=args,
                         mode=args.wb_mode)

        trainer = Trainer(data_type)

        if not trainer.is_trained():
            trainer.train()

        if trainer.is_trained():
            trainer.load_pretrained_model()

        trainer.final_eval()
        run.finish()
