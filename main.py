import os
import tempfile
import time

import albumentations as A
import cv2
import hydra
import torch
import torchvision
from ml_engine.criterion.losses import NegativeCosineSimilarityLoss
from ml_engine.criterion.simsiam import BatchWiseSimSiamLoss
from ml_engine.criterion.triplet import BatchWiseTripletLoss
from ml_engine.data.samplers import DistributedRepeatableSampler, DistributedRepeatableEvalSampler, MPerClassSampler
from ml_engine.engine import Trainer
from ml_engine.evaluation.distances import compute_distance_matrix
from ml_engine.evaluation.metrics import AverageMeter, calc_map_prak
from ml_engine.modelling.resnet import ResNetWrapper, ResNet32MixConv
from ml_engine.modelling.simsiam import SimSiamV2
from ml_engine.preprocessing.transforms import ACompose
from ml_engine.tracking.mlflow_tracker import MLFlowTracker
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from geshaem_dataset import GeshaemPatch
from transforms import CustomRandomCrop


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


class GeshaemTrainer(Trainer):
    def get_transform(self, mode, data_cfg):
        img_size = data_cfg.img_size
        if mode == 'train':
            return torchvision.transforms.Compose([
                ACompose([
                    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=15, p=0.5, value=(255, 255, 255),
                                       border_mode=cv2.BORDER_CONSTANT)
                ]),
                torchvision.transforms.RandomAffine(5, translate=(0.1, 0.1), fill=255),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0)),
                ], p=0.5),
                torchvision.transforms.Resize(img_size),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.3),
                ]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            return torchvision.transforms.Compose([
                torchvision.transforms.Resize(img_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def build_model(self, model_conf):
        if model_conf.type == 'ss2':
            model = SimSiamV2(
                arch=model_conf.arch,
                pretrained=model_conf.pretrained,
                dim=model_conf.embed_dim,
                pred_dim=model_conf.pred_dim,
                dropout=model_conf.dropout)

        elif model_conf.type == 'resnet':
            model = ResNetWrapper(
                backbone=model_conf.arch,
                weights=model_conf.pretrained,
                layers_to_freeze=model_conf.layers_freeze)

        elif model_conf.type == 'mixconv':
            model = ResNet32MixConv(
                img_size=(self._cfg.data.img_size, self._cfg.data.img_size),
                backbone=model_conf.arch,
                out_channels=model_conf.out_channels,
                mix_depth=model_conf.mix_depth,
                out_rows=model_conf.out_rows,
                weights=model_conf.pretrained,
                layers_to_freeze=model_conf.layers_freeze)

        else:
            raise NotImplementedError(f'Network {model_conf.type} is not implemented!')

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def load_dataset(self, mode, data_conf, transform):
        split = GeshaemPatch.Split.from_string(mode)
        patch_size = data_conf.img_size
        return GeshaemPatch(data_conf.path, split, transform=transform, image_size=patch_size)

    def get_dataloader(self, mode, dataset, data_conf):
        if mode in self.data_loader_registers:
            return self.data_loader_registers[mode]

        if mode == 'train':
            max_dataset_length = len(dataset) * 10
            sampler = MPerClassSampler(dataset.data_labels, m=data_conf.m_per_class,
                                       length_before_new_iter=max_dataset_length)
            dataloader = DataLoader(dataset, sampler=sampler, pin_memory=True, batch_size=data_conf.batch_size,
                                    drop_last=True, num_workers=data_conf.num_workers)
        else:
            sampler = DistributedRepeatableEvalSampler(dataset, shuffle=False, rank=self.rank,
                                                       num_replicas=self.world_size, repeat=10)

            dataloader = DataLoader(dataset, sampler=sampler, batch_size=data_conf.test_batch_size, shuffle=False,
                                    num_workers=data_conf.num_workers, pin_memory=data_conf.pin_memory, drop_last=False)

        self.data_loader_registers[mode] = dataloader
        return dataloader

    def get_criterion(self):
        if self.is_simsiam():
            return BatchWiseSimSiamLoss()
        return BatchWiseTripletLoss(margin=0.15)

    def is_simsiam(self):
        return 'ss' in self._cfg.model.type

    def validate_dataloader(self, data_loader):
        batch_time, m_ap_meter = AverageMeter(), AverageMeter()
        top1_meter, pk5_meter, pk10_meter = AverageMeter(), AverageMeter(), AverageMeter()

        end = time.time()
        embeddings, labels = [], []
        for idx, (images, targets) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=self._cfg.amp_enable):
                embs = self._model(images)
                if self.is_simsiam():
                    embs, _ = embs

            embeddings.append(embs)
            labels.append(targets)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)

        # embeddings = F.normalize(embeddings, p=2, dim=1)
        features = {}
        for feature, target in zip(embeddings, labels.numpy()):
            features.setdefault(target, []).append(feature)

        features = {k: torch.stack(v).cuda() for k, v in features.items()}
        distance_df = compute_distance_matrix(features, reduction=self._cfg.eval.distance_reduction,
                                              distance_fn=NegativeCosineSimilarityLoss())

        index_to_fragment = {i: x for i, x in enumerate(data_loader.dataset.fragments)}
        distance_df.rename(columns=index_to_fragment, index=index_to_fragment, inplace=True)

        positive_pairs = data_loader.dataset.fragment_to_group

        distance_mat = distance_df.to_numpy()
        m_ap, (top_1, prk5, prk10) = calc_map_prak(distance_mat, distance_df.columns, positive_pairs, prak=(1, 5, 10))

        m_ap_meter.update(m_ap)
        top1_meter.update(top_1)
        pk5_meter.update(prk5)
        pk10_meter.update(prk10)

        AverageMeter.reduces(m_ap_meter, top1_meter, pk5_meter, pk10_meter)

        return m_ap_meter.avg, top1_meter.avg, pk5_meter.avg, pk10_meter.avg, distance_df

    def validate_one_epoch(self, dataloader):
        m_ap, top1, pra5, pra10, distance_df = self.validate_dataloader(dataloader)
        eval_loss = 1 - m_ap

        self.log_metrics({'mAP': m_ap, 'top1': top1, 'prak5': pra5, 'prak10': pra10})
        if eval_loss < self._min_loss:
            self.log_metrics({'best_mAP': m_ap, 'best_top1': top1, 'best_prak5': pra5, 'best_prak10': pra10})
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, f'distance_matrix.csv')
                distance_df.to_csv(path)
                self._tracker.log_artifact(path, 'best_results')

        self.logger.info(f'Average: \t mAP {m_ap:.4f}\t top1 {top1:.3f}\t pr@k5 {pra5:.3f}\t pr@10 {pra10:3f}')
        return eval_loss


if __name__ == '__main__':
    dl_main()
