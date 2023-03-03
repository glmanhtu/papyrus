import logging
import math

import cv2
import numpy as np
import tkinter as tk
import torchvision.transforms

from dataset.infrared import InfraredDataset
from dataset.michigan import MichiganDataset
from options.train_options import TrainOptions


logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

args = TrainOptions(save_conf=False).parse()
transforms = torchvision.transforms.Compose([])

train_dataset = InfraredDataset(args.infrared_dir, transforms, args.image_size)
to_exclude = []

current_index = 0

while True:
    item = train_dataset[current_index]
    image = item['positive']
    # image = cv2.bitwise_not(image)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    root = tk.Tk()
    screen_h = root.winfo_screenheight()
    screen_w = root.winfo_screenwidth()
    vector = math.sqrt(0.5)
    window_h = screen_h * vector
    window_w = screen_w * vector

    if h > window_h or w > window_w:
        if h / window_h >= w / window_w:
            multiplier = window_h / h
        else:
            multiplier = window_w / w
        img = cv2.resize(img, (0, 0), fx=multiplier, fy=multiplier)

    cv2.imshow('image', img)

    # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    key = cv2.waitKey(0)
    if key == 83:
        current_index += 1
    elif key == 81:
        current_index -= 1
        if current_index < 0:
            current_index = 0
    elif key == 27:
        break
    elif key == ord('d'):
        to_exclude.append(item['pos_image'])
        print('exclude image: ' + item['pos_image'])

    # cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()

print('exclude: ')
print(to_exclude)
