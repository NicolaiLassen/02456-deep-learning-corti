import torch
import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    # log σ(Z^T . HK))
    def log_sigmoid_probs(self, z_ik, h_k):
        z_t = torch.transpose(z_ik, 0, 1)

        # Z^T . HK
        out = torch.einsum("ijk,jik->i", h_k, z_t)
        out = torch.sigmoid(out)
        return torch.log(out)

    def forward(self, z, h_k):
        # - log σ(Z . HK)) + λE [log σ(ZH . HK)]
        # TODO: ZH
        return - (self.log_sigmoid_probs(z, h_k) + self.log_sigmoid_probs(z, h_k))


if __name__ == '__main__':
    criterion = ContrastiveLoss()
    model = nn.Linear(2, 2)
    z = torch.randn(1, 10, 174, requires_grad=True)
    c = torch.randn(1, 10, 174, requires_grad=True)
    print(c.shape)
    print(z.shape)
    loss = criterion(z, c)
    print("loss", loss)
    loss.backward()
    print(model.weight.grad)
