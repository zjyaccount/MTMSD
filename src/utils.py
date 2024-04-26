import torch
from PIL import Image
import numpy as np
import random


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = acc * 100

    return acc


def oht_to_scalar(y_pred):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    return y_pred_tags
    

def colorize_mask(mask, palette):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))


def to_labels(masks, palette):
    results = np.zeros((len(masks), 256, 256), dtype=np.int32)
    label = 0
    for color in palette:
        idxs = np.where((masks == color).all(-1))
        results[idxs] = label
        label += 1
    return results
    

def setup_seed(seed):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)