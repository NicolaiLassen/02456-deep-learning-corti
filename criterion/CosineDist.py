import torch


class CosineDistLoss(torch.nn.Module):
    def __init__(self):
        super(CosineDistLoss, self).__init__()
        self.cos = torch.nn.CosineSimilarity()

    def forward(self, e, e_c):
        return (1 - self.cos(e, e_c)).mean()
