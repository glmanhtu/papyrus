from model.model_wrapper import ModelWrapper
from model.resnet18 import ResNet18


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, is_train, device, dropout=0.4):
        if args.network == 'resnet18':
            model = ResNet18(dropout=dropout)
        else:
            raise NotImplementedError(f'Model {args.network} haven\'t implemented yet!!!')

        model = ModelWrapper(args, model, is_train, device)
        return model
