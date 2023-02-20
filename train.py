import logging

from databases.michigan import MichiganDataset
from options.train_options import TrainOptions


logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

args = TrainOptions().parse()

train_dataset = MichiganDataset(args.michigan_dir, args.image_size, args.max_patches_per_fragment)
