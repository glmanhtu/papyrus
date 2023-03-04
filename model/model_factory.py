from torchvision import models

from model import simsiam
from model.model_wrapper import ModelWrapper


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, working_dir, is_train, device, dropout=0.4):
        if args.network == 'simsiam':
            model = simsiam.SimSiam(models.__dict__['resnet50'], dim=512, pred_dim=2048)
        else:
            raise NotImplementedError(f'Model {args.network} haven\'t implemented yet!!!')

        model = ModelWrapper(args, working_dir, model, is_train, device)
        return model
