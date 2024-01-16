import os
import tempfile
import time

import albumentations as A
import hydra
import torch
import torch.nn.functional as F
import torchvision
from ml_engine.criterion.losses import NegativeCosineSimilarityLoss, DistanceLoss
from ml_engine.criterion.simsiam import BatchWiseSimSiamLoss
from ml_engine.data.samplers import DistributedRepeatableEvalSampler, MPerClassSampler
from ml_engine.engine import Trainer
from ml_engine.evaluation.distances import compute_distance_matrix, compute_distance_matrix_from_embeddings
from ml_engine.evaluation.metrics import AverageMeter, calc_map_prak
from ml_engine.preprocessing.transforms import ACompose, PadCenterCrop
from ml_engine.tracking.mlflow_tracker import MLFlowTracker
from ml_engine.utils import get_combinations
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler

import transforms
import wi19_evaluate
from datasets.geshaem_dataset import GeshaemPatch, MergeDataset
from datasets.michigan_dataset import MichiganDataset


class BatchWiseTripletDistanceLoss(torch.nn.Module):

    def __init__(self, distance_fn, margin=0.15):
        super().__init__()
        self.margin = margin
        self.loss_fn = torch.nn.TripletMarginWithDistanceLoss(margin=margin, distance_function=distance_fn,
                                                              reduction='sum')

    def forward(self, samples, targets):
        n = samples.size(0)
        # split the positive and negative pairs
        eyes_ = torch.eye(n, dtype=torch.bool).cuda()
        pos_mask = targets.expand(
            targets.shape[0], n
        ).t() == targets.expand(n, targets.shape[0])
        neg_mask = ~pos_mask
        pos_mask[:, :n] = pos_mask[:, :n] * ~eyes_

        pos_groups, neg_groups = [], []
        for i in range(n):
            it = torch.tensor([i], device=samples.device)
            pos_pair_idx = torch.nonzero(pos_mask[i, i:]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                pos_combinations = get_combinations(it, pos_pair_idx + i)

                neg_pair_idx = torch.nonzero(neg_mask[i, :]).view(-1)
                if pos_combinations.shape[0] > 0 and neg_pair_idx.shape[0] > 0:
                    neg_combinations = get_combinations(it, neg_pair_idx)

                    if pos_combinations.shape[0] < neg_combinations.shape[0]:
                        neg_target_distance = targets[neg_combinations[:, 0]] - targets[neg_combinations[:, 1]]
                        argsort = torch.argsort(torch.abs(neg_target_distance), descending=True)
                        n_negs = int(0.9 * neg_combinations.shape[0])
                        neg_combinations = neg_combinations[argsort][:n_negs]
                        if n_negs > pos_combinations.shape[0]:
                            pos_combinations = pos_combinations[torch.randint(high=pos_combinations.shape[0],
                                                                              size=(n_negs,))]
                    elif neg_combinations.shape[0] < 1:
                        continue
                    elif neg_combinations.shape[0] < pos_combinations.shape[0]:
                        neg_combinations = neg_combinations[torch.randint(high=neg_combinations.shape[0],
                                                                          size=(pos_combinations.shape[0],))]
                    neg_groups.append(neg_combinations)
                    pos_groups.append(pos_combinations)

        pos_groups = torch.cat(pos_groups, dim=0)
        neg_groups = torch.cat(neg_groups, dim=0)

        assert torch.equal(pos_groups[:, 0], neg_groups[:, 0])
        anchor = samples[pos_groups[:, 0]]
        positive = samples[pos_groups[:, 1]]
        negative = samples[neg_groups[:, 1]]

        return self.loss_fn(anchor, positive, negative)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def dl_main(cfg: DictConfig):
    tracker = MLFlowTracker(cfg.exp.name, cfg.exp.tracking_uri, tags=cfg.exp.tags)
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


def distance_fn(x, y):
    return 1.0 - F.cosine_similarity(x, y, dim=-1)


class GeshaemTrainer(Trainer):
    def get_transform(self, mode, data_cfg):
        img_size = data_cfg.img_size
        if mode == 'train':
            return torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(img_size, pad_if_needed=True, fill=(255, 255, 255)),
                ACompose([
                    A.CoarseDropout(max_holes=32, min_holes=3, min_height=16, max_height=64, min_width=16, max_width=64,
                                    fill_value=255, p=0.9),
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
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def load_dataset(self, mode, data_conf, transform):
        if data_conf.name == 'geshaem':
            split = GeshaemPatch.Split.from_string(mode)
            return GeshaemPatch(data_conf.path, split, transform=transform,
                                include_verso=data_conf.include_verso)
        elif data_conf.name == 'michigan':
            return MichiganDataset(data_conf.path, MichiganDataset.Split.from_string(mode), transform)
        elif data_conf.name == 'merge':
            if mode == 'train':
                michigan = MichiganDataset(data_conf.path_michigan, MichiganDataset.Split.from_string(mode), transform)
                geshaem = GeshaemPatch(data_conf.path_geshaem,  GeshaemPatch.Split.from_string(mode),
                                       transform=transform, include_verso=data_conf.include_verso,
                                       base_idx=len(michigan.labels))
                return MergeDataset([michigan, geshaem], transform)
            else:
                return GeshaemPatch(data_conf.path_geshaem,  GeshaemPatch.Split.from_string(mode), transform=transform,
                                    include_verso=data_conf.include_verso)
        else:
            raise NotImplementedError(f'Dataset {data_conf.name} not implemented!')

    def get_dataloader(self, mode, dataset, data_conf):
        if mode in self.data_loader_registers:
            return self.data_loader_registers[mode]

        if mode == 'train':
            if data_conf.m_per_class == 0:
                sampler = RandomSampler(dataset)
                sampler.set_epoch = lambda x: x
            else:
                max_dataset_length = len(dataset) * data_conf.m_per_class
                sampler = MPerClassSampler(dataset.data_labels, m=data_conf.m_per_class,
                                           length_before_new_iter=max_dataset_length)
            dataloader = DataLoader(dataset, sampler=sampler, pin_memory=True, batch_size=data_conf.batch_size,
                                    drop_last=True, num_workers=data_conf.num_workers)
        else:
            sampler = DistributedRepeatableEvalSampler(dataset, shuffle=False, rank=self.rank,
                                                       num_replicas=self.world_size, repeat=1)

            dataloader = DataLoader(dataset, sampler=sampler, batch_size=data_conf.test_batch_size, shuffle=False,
                                    num_workers=data_conf.num_workers, pin_memory=data_conf.pin_memory, drop_last=False)

        self.data_loader_registers[mode] = dataloader
        return dataloader

    def get_criterion(self):
        if self.is_simsiam():
            return DistanceLoss(BatchWiseSimSiamLoss(), NegativeCosineSimilarityLoss(reduction='none'))
        elif self.is_classifier():
            return DistanceLoss(torch.nn.CrossEntropyLoss(), distance_fn=distance_fn)
        return DistanceLoss(BatchWiseTripletDistanceLoss(margin=self._cfg.train.triplet_margin, distance_fn=distance_fn),
                            distance_fn=distance_fn)

    def is_simsiam(self):
        return 'ss2' in self._cfg.model.type

    def is_classifier(self):
        return 'classifier' in self._cfg.model.type

    def validate_one_epoch(self, dataloader):
        batch_time = AverageMeter()
        loss_meter = AverageMeter()

        criterion = self.get_criterion()
        end = time.time()
        embeddings, labels = [], []
        for idx, (images, targets) in enumerate(dataloader):
            images = images.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=self._cfg.amp_enable):
                embs = self._model(images)
                if self.is_simsiam():
                    embs, _ = embs
            loss = criterion(embs, targets.cuda())
            loss_meter.update(loss.cpu().item(), images.size(0))

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

            positive_pairs = dataloader.dataset.fragment_to_group

            distance_mat = distance_df.to_numpy()
            m_ap, (top1, pra5, pra10) = calc_map_prak(distance_mat, distance_df.columns, positive_pairs, prak=(1, 5, 10))

        eval_loss = 1 - m_ap

        self.log_metrics({'Loss': loss_meter.avg, 'mAP': m_ap, 'top1': top1, 'prak5': pra5, 'prak10': pra10})
        if eval_loss < self._min_loss:
            self.log_metrics({'best_mAP': m_ap, 'best_top1': top1, 'best_prak5': pra5, 'best_prak10': pra10})
            if distance_df is not None:
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, f'distance_matrix.csv')
                    distance_df.to_csv(path)
                    self._tracker.log_artifact(path, 'best_results')

        self.logger.info(f'Average: \t loss: {loss_meter.avg:.4f}\t mAP {m_ap:.4f}\t top1 {top1:.3f}\t'
                         f' pr@k5 {pra5:.3f}\t pr@10 {pra10:3f}')
        return eval_loss


if __name__ == '__main__':
    dl_main()
