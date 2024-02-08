import os
import time
from functools import lru_cache

import albumentations as A
import hydra
import torch
import torch.nn.functional as F
import torchvision
from ml_engine.criterion.losses import DistanceLoss
from ml_engine.criterion.triplet import BatchWiseTripletDistanceLoss
from ml_engine.data.samplers import MPerClassSampler, DistributedSamplerWrapper
from ml_engine.engine import Trainer
from ml_engine.evaluation.distances import compute_distance_matrix, compute_distance_matrix_from_embeddings
from ml_engine.evaluation.metrics import AverageMeter, calc_map_prak
from ml_engine.preprocessing.transforms import ACompose, PadCenterCrop
from ml_engine.tracking.mlflow_tracker import MLFlowTracker
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler

import transforms
import wi19_evaluate
from datasets.geshaem_dataset import GeshaemPatch
from datasets.geshaem_dataset_v2 import GeshaemPatchV2
from datasets.michigan_dataset import MichiganDataset


@hydra.main(version_base=None, config_path="conf", config_name="config")
def dl_main(cfg: DictConfig):
    tracker = MLFlowTracker(cfg.exp.name, cfg.exp.tracking_uri, rank=0, tags=cfg.exp.tags)
    trainer = GeshaemTrainer(cfg, tracker)
    with tracker.start_tracking(run_id=cfg.run.run_id, run_name=cfg.run.name, tags=dict(cfg.run.tags)):
        if cfg.mode == 'eval':
            trainer.validate()
        elif cfg == 'throughput':
            trainer.throughput()
        else:
            trainer.train()

        exp_log_dir = os.path.join(cfg.log_dir, cfg.run.name)
        tracker.log_artifacts(exp_log_dir, 'logs')
    del trainer
    del tracker


def distance_fn(x, y):
    return 1.0 - F.cosine_similarity(x, y, dim=-1)


class GeshaemTrainer(Trainer):
    def get_transform(self, mode, data_cfg):
        img_size = data_cfg.img_size
        if mode == 'train':
            return torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(img_size, pad_if_needed=True, fill=(255, 255, 255)),
                torchvision.transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                ACompose([
                    A.CoarseDropout(max_holes=16, min_holes=1, min_height=16, max_height=64, min_width=16, max_width=64,
                                    fill_value=255, p=0.8),
                ]),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1),
                ], p=.5),
                transforms.GaussianBlur(p=0.5, radius_max=1),
                # transforms.Solarization(p=0.2),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            return torchvision.transforms.Compose([
                PadCenterCrop((img_size, img_size), pad_if_needed=True, fill=(255, 255, 255)),
                torchvision.transforms.Resize(int(img_size * 1.2)),
                torchvision.transforms.CenterCrop(img_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def load_dataset(self, mode, data_conf, transform):
        if data_conf.name == 'geshaem' and data_conf.version == 1:
            split = GeshaemPatch.Split.from_string(mode)
            return GeshaemPatch(data_conf.path, split, transform=transform, include_verso=data_conf.include_verso)
        elif data_conf.name == 'geshaem' and data_conf.version == 2:
            split = GeshaemPatchV2.Split.from_string(mode)
            return GeshaemPatchV2(data_conf.path, split, transform=transform, include_verso=data_conf.include_verso)
        elif data_conf.name == 'michigan':
            return MichiganDataset(data_conf.path, MichiganDataset.Split.from_string(mode), transform)
        else:
            raise NotImplementedError(f'Dataset {data_conf.name} not implemented!')

    @lru_cache
    def get_dataloader(self, mode, dataset, data_conf):
        if mode == 'train':
            if data_conf.m_per_class == 0:
                sampler = RandomSampler(dataset)
                sampler.set_epoch = lambda x: x
            else:
                max_dataset_length = len(dataset) * data_conf.data_repeat
                sampler = MPerClassSampler(dataset.data_labels, m=data_conf.m_per_class,
                                           length_before_new_iter=max_dataset_length)
                sampler = DistributedSamplerWrapper(sampler, self.world_size, self.rank, shuffle=False)
            dataloader = DataLoader(dataset, sampler=sampler, pin_memory=True, batch_size=data_conf.batch_size,
                                    drop_last=True, num_workers=data_conf.num_workers)
        else:
            # We don't apply distributed sampler here because the validating set is not large
            dataloader = DataLoader(dataset, batch_size=data_conf.test_batch_size, shuffle=False,
                                    num_workers=data_conf.num_workers, pin_memory=data_conf.pin_memory, drop_last=False)

        return dataloader

    def get_criterion(self):
        return DistanceLoss(
            BatchWiseTripletDistanceLoss(distance_fn, margin=self._cfg.train.triplet_margin, reduction='sum'),
            distance_fn=distance_fn
        )

    def output_mapping(self, output):
        return output

    def validate_one_epoch(self, dataloader):
        batch_time = AverageMeter()

        criterion = self.get_criterion()
        end = time.time()
        embeddings, labels = [], []
        for idx, (images, targets) in enumerate(dataloader):
            images = images.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=self._cfg.amp_enable):
                embs = self._model(images)
                embs = self.output_mapping(embs)

            embeddings.append(embs)
            labels.append(targets)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)
        self.logger.info(f'N samples: {len(embeddings)}, N categories: {len(torch.unique(labels))}')

        if self._cfg.data.name == 'michigan':
            distance_matrix = compute_distance_matrix_from_embeddings(embeddings, criterion.compute_distance)
            m_ap, top1, pra5, pra10 = wi19_evaluate.get_metrics(distance_matrix.numpy(), labels.numpy())
            distance_df = None
        else:
            # embeddings = F.normalize(embeddings, p=2, dim=1)
            features = {}
            for feature, target in zip(embeddings, labels.numpy()):
                features.setdefault(target, []).append(feature)

            features = {k: torch.stack(v).cuda() for k, v in features.items()}
            distance_df = compute_distance_matrix(features, reduction=self._cfg.eval.distance_reduction,
                                                  distance_fn=criterion.compute_distance)

            index_to_fragment = {i: x for i, x in enumerate(dataloader.dataset.fragments)}
            distance_df.rename(columns=index_to_fragment, index=index_to_fragment, inplace=True)

            pos_pairs = dataloader.dataset.fragment_to_group

            distance_mat = distance_df.to_numpy()
            m_ap, (top1, pra5, pra10) = calc_map_prak(distance_mat, distance_df.columns, pos_pairs, prak=(1, 5, 10))

        eval_loss = 1 - m_ap

        self.log_metrics({'mAP': m_ap, 'top1': top1, 'prak5': pra5, 'prak10': pra10})
        if eval_loss < self._min_loss:
            self.log_metrics({'best_mAP': m_ap, 'best_top1': top1, 'best_prak5': pra5, 'best_prak10': pra10})
            if distance_df is not None:
                self._tracker.log_table_as_csv(distance_df, 'best_results', 'distance_matrix.csv')

        self.logger.info(f'Average: \t mAP {m_ap:.4f}\t top1 {top1:.3f}\t pr@k5 {pra5:.3f}\t pr@10 {pra10:3f}')
        return eval_loss


if __name__ == '__main__':
    dl_main()
