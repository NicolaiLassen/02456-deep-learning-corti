import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch


class Wav2vec(nn.Module):

    def __init__(self):
        super(Wav2vec, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        activation = nn.ReLU()
        dropout = 0.0
        self.encoder = Encoder(activation, dropout)
        self.context = Context()

    def forward(self, x):
        z = self.encoder(x)
        #c = self.context(x)
        # x = x.view(-1, self.num_flat_features(x))
        return z#, c


class Encoder(nn.Module):
    def __init__(self, activation, dropout):
        super(Encoder, self).__init__()

        self.in_c = 50
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=self.in_c, kernel_size=10, stride=5),
                                     nn.Dropout(p=dropout),
                                     nn.GroupNorm(1, self.in_c),  # Affine, what to do?
                                     activation,
                                     # 2nd layer
                                     nn.Conv1d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=8, stride=4),
                                     nn.Dropout(p=dropout),
                                     ## See norm_block - FB_repo
                                     nn.GroupNorm(1, self.in_c),  # Affine, what to do?
                                     activation,
                                     # 3rd layer
                                     nn.Conv1d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=4, stride=2),
                                     nn.Dropout(p=dropout),
                                     nn.GroupNorm(1, self.in_c),  # Affine, what to do?
                                     activation,
                                     # Fourth layer
                                     nn.Conv1d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=4, stride=2),
                                     nn.Dropout(p=dropout),
                                     nn.GroupNorm(1, self.in_c),  # Affine, what to do?
                                     activation,
                                     # Fifth layer
                                     nn.Conv1d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=4, stride=2),
                                     nn.Dropout(p=dropout),
                                     nn.GroupNorm(1, self.in_c),  # Affine, what to do?
                                     activation)

    def log_compression(self, x):
        # https://www.edn.com/log-1-x-compression/
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


class ZeroPad1d(nn.Module):
    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))


if __name__ == '__main__':
    # TODO: Test the network
    import matplotlib.pyplot as plt


    def plot_wav(waveform):
        plt.figure(1)
        plt.title("Test wave")
        plt.plot(waveform[0])
        plt.show()


    waveform, sample_rate = torchaudio.load("wav_16k_example.wav")
    #torch.unsqueeze(waveform, 1)
    print(waveform.shape)
    #plot_wav(waveform)
    model = Wav2vec()
    # For testing unsqueeze to match conv1d shape requirements
    out = model(torch.unsqueeze(waveform, 1))
    #print(waveform)
