from typing import List

import torch
import torch.nn.functional as F
from torch import topk


def collate(batch) -> (torch.Tensor, List[str]):
    max_l = max([x[0].shape[1] for x in batch])
    wav = torch.cat([F.interpolate(data[0].unsqueeze(0), size=(max_l)) for data in batch], dim=0)
    text = [data[2].lower() for data in batch]
    return wav, text


def GreedyDecoder(ctc_matrix, blank_label=0):
    """Greedy Decoder. Returns highest probability of
        class labels for each timestep
        # TODO: collapse blank labels
    Args:
        ctc_matrix (torch.Tensor):
            shape (1, num_classes, output_len)
        blank_label (int): blank labels to collapse

    Returns:
        torch.Tensor: class labels per time step.
         shape (ctc timesteps)
    """
    top = topk(ctc_matrix, k=1, dim=1)
    return top[1][0][0]
