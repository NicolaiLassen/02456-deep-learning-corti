import torch


# Ref: https://arxiv.org/pdf/1909.08782.pdf
# https://github.com/Leethony/Additive-Margin-Softmax-Loss-Pytorch
class MaskedMarginSoftmax(torch.nn.Module):
    def __init__(self):
        super(MaskedMarginSoftmax, self).__init__()

    # -1/B sum_i=1^B log (e^{z_ii-delta} / e^{z_ii-delta} + sum_j=1^B M_jj e^{Z_ij}
    def log_batch_retrievals(self, x, y):
        return 0

    def forward(self, z, m):
        # L_xy + L_yx
        return self.log_batch_retrievals(z, m) + self.log_batch_retrievals(z, m)
