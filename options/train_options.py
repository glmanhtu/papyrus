from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def is_train(self):
        return True

    def __init__(self):
        super().__init__()

    def initialize(self):
        BaseOptions.initialize(self)

