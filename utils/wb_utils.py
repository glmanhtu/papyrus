import numpy as np
import seaborn as sns
import wandb
from matplotlib import pyplot as plt

from utils.transform import reverse_transform

chart_limit = wandb.Table.MAX_ROWS


def wb_img(image):
    return wandb.Image(reverse_transform()(image))


def create_similarity_heatmap(similarity_matrix):
    # Create the heatmap using seaborn
    heatmap = sns.heatmap(similarity_matrix, cmap="YlGnBu")

    # Get the figure object from the heatmap
    fig = heatmap.get_figure()

    # Convert the figure to a numpy array
    fig.canvas.draw()
    heatmap_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return heatmap_array