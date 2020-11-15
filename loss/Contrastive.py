import torch
import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    # W_k C_i + b_k
    def h_k(self, c_i):
        return torch.einsum("w,c->wc", c_i, c_i) + c_i

    # log σ(z_i+k h_k(c_i))
    def log_sig_probs(self, z_ik, c_i):
        z_i_k_t = torch.transpose(z_ik, 0, 1)
        h_k_c_i = self.h_k(c_i)
        out = torch.add(z_i_k_t, h_k_c_i)
        out = torch.sigmoid(out)
        return torch.log(out)

    # keep torch grad
    def sum_pass(self, i, k, z, c):
        if i == k:
            return 0
        return torch.add(self.log_sig_probs(z[i + k], c[i]) + 0, self.sum_pass(i + 1, k, z, c))

    # keep torch grad
    def cat_pass(self, T, k, z, c):
        if k == 0:
            return self.sum_pass(T - k, k, z, c)
        return self.sum_pass(T - k, k, z, c)

    def forward(self, z, c, T=3, k=3):
        # - sum_i=1^T-k
        return self.cat_pass(T, k, z, c)


if __name__ == '__main__':
    criterion = ContrastiveLoss()
    model = nn.Linear(2, 2)
    z = torch.randn(6, 1, 2)
    c = torch.randn(6, 1, 2)
    z = model(z)
    c = model(c)
    print(z)
    loss = criterion(z, c)
    print("loss", loss)
    loss.backward()
    print(model.weight.grad)
