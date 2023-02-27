import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import pyplot as plt

from utils.misc import add_description
from utils.transform import reverse_transform

chart_limit = wandb.Table.MAX_ROWS


def wb_img(image):
    return wandb.Image(reverse_transform()(image))


def generate_query_table(query_result, top_k=25):
    columns = ["Desc"] + [f'query #{i + 1}' for i in range(len(query_result))]
    data = []

    record = ['Query img']
    for query in query_result:
        img = add_description(query['query_img'], 'Similarity: - ', query['query'])
        record.append(wandb.Image(img))
    data.append(record)

    for idx in range(top_k):
        record = [f'#{idx + 1}']
        for query in query_result:
            target = query['results'][idx]
            bottom_desc = 'Similarity {:.4f}'.format(target['similarity'])
            img = add_description(target['target_img'], bottom_desc, target['target'])
            record.append(wandb.Image(img))
        data.append(record)
    return wandb.Table(data=data, columns=columns)


def create_heatmap(similarity_matrix: pd.DataFrame, dpi=200):
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
