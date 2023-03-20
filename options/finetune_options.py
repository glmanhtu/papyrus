from .base_options import BaseOptions


class FinetuneOptions(BaseOptions):
    def is_train(self):
        return True

    def __init__(self, save_conf=True):
        super().__init__(save_conf)

    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--pretrained_model', type=str, help='Path to the pretrained model', required=True)


