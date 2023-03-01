from torchvision import models

from model import simsiam
from model.model_wrapper import ModelWrapper
from model.resnet18 import ResNet18
from model.resnet50 import ResNet50


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, working_dir, is_train, device, dropout=0.4):
        if args.network == 'resnet18':
            model = ResNet18(dropout=dropout)
        elif args.network == 'resnet50':
            model = ResNet50(dropout=dropout)
        elif args.network == 'simsiam':
            model = simsiam.SimSiam(models.__dict__['resnet50'], dim=128, pred_dim=512)
        else:
            raise NotImplementedError(f'Model {args.network} haven\'t implemented yet!!!')

        model = ModelWrapper(args, working_dir, model, is_train, device)
        return model
