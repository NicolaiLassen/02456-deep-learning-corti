import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch
import math


class Wav2vec(nn.Module):

    def __init__(self):
        super(Wav2vec, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        activation = nn.ReLU()
        dropout = 0.0
        self.encoder = Encoder(activation, dropout)
        self.context = Context(10, 10, 3, 0.5, nn.ReLU())

        # Calculate offset for prediction module
        # NOT SURE THAT WE NEED THIS?!
        def calc_offset():
            jin = 0
            rin = 0
            for layer in next(self.encoder.children()):
                if layer.__class__.__name__ == 'Conv1d':
                    k = layer.kernel_size[0]
                    stride = layer.stride[0]
                    if rin == 0:
                        rin = k
                    rin = rin + (k - 1) * jin
                    if jin == 0:
                        jin = stride
                    else:
                        jin *= stride
            offset = math.ceil(rin / jin)

            return int(offset)

        self.offset = calc_offset()

    def forward(self, x):
        z = self.encoder(x)
        c = self.context(z)
        # x = x.view(-1, self.num_flat_features(x))
        return z, c


class Encoder(nn.Module):
    def __init__(self, activation, dropout):
        super(Encoder, self).__init__()
        self.in_c = 10

        # Hardcoded architecture, as the blocks are different
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
    def __init__(self, n_in, n_out, k, dropout, activation, layers=10):
        super(Context, self).__init__()

        # All block are the same, so create using a function
        def conv_block(n_in, n_out, k, dropout, activation):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, padding=1),
                nn.Dropout(p=dropout),
                nn.GroupNorm(1, n_out),
                activation
            )

        # Holder for conv layers
        self.conv = nn.ModuleList()

        # Create #layers number of conv-blocks
        for i in range(0, layers):
            self.conv.append(conv_block(n_in, n_out, k, dropout, activation))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        x = self.conv(x)
        return x


class Prediction(nn.Module):
    def __init__(self, predictions_steps):
        in_dim = 10  # antal channels fra aggregator UPDATE TO USE AS ARGUMENT IN CONSTRUCTOR
        out_dim = 10  # antal outchannels fra encoder UPDATE TO USE AS ARGUMENT IN CONSTRUCTOR
        prediction_steps = 12  # Not sure what this is? It's an argument UPDATE TO USE AS ARGUMENT IN CONSTRUCTOR
        super(Prediction, self).__init__()
        self.transpose_context = nn.ConvTranspose2d(in_dim, out_dim, (1, prediction_steps))
        self.sample_distance = None
        self.n_negatives = 1

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)

        cross_high = tsz * bsz
        high = tsz if self.sample_distance is None else min(tsz, self.sample_distance)
        assert high > 1

        neg_idxs = torch.randint(low=0, high=high, size=(bsz, self.n_negatives * tsz))

        with torch.no_grad():
            if self.n_negatives > 0:
                tszs = (
                    # REMEMBER TO INCLUDE BUFFERED_ARANGE FROM UTIL
                    buffered_arange(tsz)
                        .unsqueeze(-1)
                        .expand(-1, self.n_negatives)
                        .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * tsz)
                )
                neg_idxs[neg_idxs >= tszs] += 1

        for i in range(1, bsz):
            neg_idxs[i] += i * high

        negs = y[..., neg_idxs.view(-1)]
        negs = negs.view(
            fsz, bsz, self.n_negatives + 0, tsz
        ).permute(
            2, 1, 0, 3
        )  # to NxBxCxT
        return negs

    def forward(self, c, z):
        c = c.unsqueeze(-1)
        # Transpose to give steps predictions into the future
        c = self.transpose_context(c)


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
