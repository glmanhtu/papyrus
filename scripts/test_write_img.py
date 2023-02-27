import numpy as np
import matplotlib

from utils.misc import add_description

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

img = np.zeros([224, 224, 3], dtype=np.uint8)

img = add_description(img, 'Similarity: 0.9814', 'IMGN_ADJ_FDSS')

dpi = 80
height, width, depth = img.shape

# What size does the figure need to be in inches to fit the image?
figsize = width / float(dpi), height / float(dpi)

# Create a figure of the right size with one axes that takes up the full figure
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0, 0, 1, 1])

# Hide spines, ticks, etc.
ax.axis('off')

# Display the image.
ax.imshow(img, cmap='gray')

plt.show()
