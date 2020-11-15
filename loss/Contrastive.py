import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    # W_k C_i + b_k
    def h_k(self, c_i):
        return W_k * c_i + b_k

    # log σ(z_i+k h_k(c_i))
    def log_sig_probs(self, z_ik, c_i):
        z_i_k_t = torch.transpose(z_ik, 0, 1)
        sigma = F.sigmoid(z_i_k_t * self.h_k(c_i))
        return torch.log(sigma)

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, z, c):
        # self.check_type_forward((x0, x1, y))

        # - sum_i=1^T-k
        loss = 0
        for i in range(1, T - k):
            loss += self.log_sig_probs(z[i + k]) + self.log_sig_probs(?s)


        return loss
