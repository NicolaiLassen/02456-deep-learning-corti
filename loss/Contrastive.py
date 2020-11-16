import torch
import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    # W_k . C_i + b_k
    def h_k(self, c_i):
        print(c_i)
        # get W_k
        # b_k
        return  # h_k

    # log σ(z_i+k^T . h_k(c_i))
    def log_sig_probs(self, z_ik, c_i):
        # z_ik^T
        z_i_k_t = torch.transpose(z_ik, 0, 1)
        h_k_c_i = self.h_k(c_i)

        #
        out = torch.einsum("", z_i_k_t, h_k_c_i)
        out = torch.sigmoid(out)
        return torch.log(out)

    # keep torch grad
    def sum_pass(self, T, k, z, c, i):
        # k time steps probs
        # probs_1 + ... + probs_i-k

        if i == T - k:
            return self.sum_pass(T, k, z, c, i + 1)

        # log σ(z_i+k^T . h_k(c_i)) + lambda E [log σ(˜z^T . h_k(c_i))]
        z_tilde = torch.randn(z[i + k].shape)
        z_c_probs = self.log_sig_probs(z[i + k], c[i])
        z_tilde_c_probs = self.log_sig_probs(z_tilde, c[i])
        probs = torch.add(z_c_probs, z_tilde_c_probs)

        return torch.add(probs, self.sum_pass(T, k, z, c, i + 1))

    # keep torch grad
    def cat_pass(self, T, k, z, c):
        # los for each step
        # [L_k...L_K]

        if k == 0:
            return self.sum_pass(T - k, k, z, c, 1)

        return torch.cat(
            self.sum_pass(T - k, k, z, c, 1),
            self.cat_pass(T - k, k - 1, z, c)
        )

    def forward(self, z, c, T=3, k=3):
        # - sum_i=1^T-k
        return self.cat_pass(T, k, z, c)


if __name__ == '__main__':
    criterion = ContrastiveLoss()
    model = nn.Linear(2, 2)
    z = torch.randn(1, 10, 174, requires_grad=True)
    c = torch.randn(1, 10, 174, requires_grad=True)
    z = model(z)
    c = model(c)
    print(c.shape)
    print(z.shape)
    loss = criterion(z, c)
    print("loss", loss)
    loss.backward()
    print(model.weight.grad)
