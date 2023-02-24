import numpy as np
import seaborn as sns
import wandb
from matplotlib import pyplot as plt

from utils.transform import reverse_transform

chart_limit = wandb.Table.MAX_ROWS


def wb_img(image):
    return wandb.Image(reverse_transform()(image))


def create_distance_heatmap(similarity_matrix, dpi=200):
    # Create the heatmap using seaborn
    heatmap = sns.heatmap(similarity_matrix, cmap="YlGnBu")

    # Get the figure object from the heatmap
    fig = heatmap.get_figure()

    # Set the DPI of the matplotlib figure
    plt.figure(dpi=dpi)

    # Convert the figure to a numpy array
    fig.canvas.draw()
    heatmap_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Close the figure to free up memory
    plt.close(fig)

    return heatmap_array
