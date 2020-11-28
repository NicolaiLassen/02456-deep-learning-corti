import torch
import torch.nn.functional as F

def collate(batch):
    max_l = max([x[0].shape[1] for x in batch])
    d = [(F.interpolate(data[0].unsqueeze(0), size=(max_l)).squeeze(0), data[2].lower()) for data in batch]
    return d