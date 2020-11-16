import torch
import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    # log σ(X^T . Y))
    def log_sigmoid_probs(self, x, y):
        x_t = torch.transpose(x, 0, 1)
        # Z^T . HK
        out = torch.einsum("jik,ijk->i", x_t, y)
        out = torch.sigmoid(out)
        return torch.log(out)

    def forward(self, z, z_n, h_k):
        # - log σ(Z . HK)) + λE [log σ(ZN . HK)]
        return - (self.log_sigmoid_probs(z, h_k) + self.log_sigmoid_probs(z_n, h_k))


if __name__ == '__main__':
    criterion = ContrastiveLoss()
    model = nn.Linear(2, 2)
    z = torch.randn(1, 10, 174, requires_grad=True)
    zn = torch.randn(1, 10, 174, requires_grad=True)
    c = torch.randn(1, 10, 174, requires_grad=True)
    print(c.shape)
    print(z.shape)
    loss = criterion(z, zn, c)
    print("loss", loss)
    loss.backward()
    print(model.weight.grad)
