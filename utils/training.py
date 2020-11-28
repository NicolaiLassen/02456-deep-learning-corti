import torch
import torch.nn.functional as F
from typing import List


def collate(batch) -> (torch.Tensor, List[str]):
    max_l = max([x[0].shape[1] for x in batch])
    wav = torch.cat([F.interpolate(data[0].unsqueeze(0), size=(max_l)) for data in batch], dim = 0)
    text = [data[2].lower() for data in batch]
    return wav, text