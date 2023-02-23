import torch
import wandb
from torch import nn

from dataset.utils import idx_to_letter
from utils.transform import reverse_transform


def wb_img(image):
    return wandb.Image(reverse_transform()(image))


def log_prediction(wb_table, columns, batch_data, anchor_out, pos_out,
                   neg_out, n_items, bin_weight):
    res = {'id': [x for x in range(n_items)]}
    res['anchor'] = batch_data['img_anchor'][:n_items].cpu()
    if 'anchor_bin' in columns:
        res['anchor_bin_pred'] = anchor_out['reconstruct'][:n_items].cpu() / bin_weight
        res['anchor_bin'] = batch_data['bin_anchor'][:n_items].cpu()
    res['positive'] = batch_data['img_positive'][:n_items].cpu()
    res['negative'] = batch_data['img_negative'][:n_items].cpu()
    if 'symbol' in columns:
        res['symbol'] = batch_data['symbol'][:n_items].cpu().numpy()
        res['symbol_pred'] = torch.max(anchor_out['symbol'][:n_items], dim=1).indices.cpu().numpy()
    distance_func = nn.MSELoss(reduction='none')
    res['pos_distance'] = distance_func(anchor_out['footprint'][:n_items], pos_out['footprint'][:n_items]).mean(dim=1)
    res['neg_distance'] = distance_func(anchor_out['footprint'][:n_items], neg_out['footprint'][:n_items]).mean(dim=1)

    _id = 0
    key_img_type = {'anchor', 'anchor_bin', 'anchor_bin_pred', 'positive', 'negative'}

    for i in range(n_items):
        params = []
        for key in columns:
            val = res[key][i]
            if key in key_img_type:
                val = wb_img(val)
            if key == 'symbol' or key == 'symbol_pred':
                val = idx_to_letter[val]
            params.append(val)
        wb_table.add_data(*params)
