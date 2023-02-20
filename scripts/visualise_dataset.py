import logging
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms

from databases.michigan import MichiganDataset
from options.train_options import TrainOptions


logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

args = TrainOptions().parse()
transforms = torchvision.transforms.Compose([])

train_dataset = MichiganDataset(args.michigan_dir, transforms, args.image_size)

for item in train_dataset:
    plt.figure()
    image = np.concatenate([item['positive'], item['anchor'], item['negative']], axis=0)
    plt.imshow(image)
    plt.show()  # display it