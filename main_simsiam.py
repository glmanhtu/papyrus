import os

import hydra
from ml_engine.criterion.losses import NegativeCosineSimilarityLoss, DistanceLoss
from ml_engine.criterion.simsiam import BatchWiseSimSiamLoss
from ml_engine.tracking.mlflow_tracker import MLFlowTracker
from omegaconf import DictConfig

from main import GeshaemTrainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def dl_main(cfg: DictConfig):
    tracker = MLFlowTracker(cfg.exp.name, cfg.exp.tracking_uri, rank=0, tags=cfg.exp.tags)
    trainer = GeshaemSimsiamTrainer(cfg, tracker)
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


class GeshaemSimsiamTrainer(GeshaemTrainer):
    def get_criterion(self):
        return DistanceLoss(BatchWiseSimSiamLoss(), NegativeCosineSimilarityLoss(reduction='none'))

    def output_mapping(self, output):
        embs, _ = output
        return embs


if __name__ == '__main__':
    dl_main()
