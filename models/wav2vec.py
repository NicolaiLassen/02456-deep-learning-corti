import torch
import torch.nn as nn
import torch.nn.functional as F


class Wav2vec(nn.Module):

    def __init__(self):
        super(Wav2vec, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.encoder = Encoder()


    def forward(self, x):
        x = self.encoder(x)

        #x = x.view(-1, self.num_flat_features(x))

        return x


class Encoder(nn.Module):
    def __init__(self, activation, p):
        super(Encoder, self).__init__()
        self.in_c = 50

        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=self.in_c, kernel_size=10, stride=5),
                                     nn.Dropout(p=p),
                                     nn.GroupNorm(1, dim=self.in_c), # Affine, what to do?
                                     activation,
                                     # 2nd layer
                                     nn.Conv1d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=8, stride=4),
                                     nn.Dropout(p=p),
                                     nn.GroupNorm(1, dim=self.in_c),  # Affine, what to do?
                                     activation,
                                     # 3rd layer
                                     nn.Conv1d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=4, stride=2),
                                     nn.Dropout(p=p),
                                     nn.GroupNorm(1, dim=self.in_c),  # Affine, what to do?
                                     activation,
                                     # Fourth layer
                                     nn.Conv1d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=4, stride=2),
                                     nn.Dropout(p=p),
                                     nn.GroupNorm(1, dim=self.in_c),  # Affine, what to do?
                                     activation,
                                     # Fifth layer
                                     nn.Conv1d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=4, stride=2),
                                     nn.Dropout(p=p),
                                     nn.GroupNorm(1, dim=self.in_c),  # Affine, what to do?
                                     activation)

    def log_compression(self, x):
        x = x.abs()
        x = x + 1
        return x.log()

    def forward(self, x):
        x = self.encoder(x)
        x = self.log_compression(x)

        # TODO implement skipped connections?
        return x

class Context(nn.Module):
    def __init__(self):
        super(Context, self).__init__()


