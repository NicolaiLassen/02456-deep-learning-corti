import torch
import torch.nn as nn


## SIMPLE DIST LOSS
class DistLoss(torch.nn.Module):
    def __init__(self):
        super(DistLoss, self).__init__()
        self.cos = nn.CosineSimilarity()

    def forward(self, e, e_c):
        # - (log σ(Z^T . HK)) + λE [log σ(ZN^T . HK)])
        return (1 - self.cos(e, e_c)).mean()
