import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def log_compression(x):
    # https://www.edn.com/log-1-x-compression/
    x = x.abs()
    x = x + 1
    return x.log()


class Wav2vec(nn.Module):

    def __init__(self):
        super(Wav2vec, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel

        channels = 512
        activation = nn.ReLU()
        dropout = 0.1

        self.encoder = Encoder(channels=channels, activation=activation, dropout=dropout)
        self.context = Context(channels=channels, k=3, dropout=dropout, activation=activation)
        self.prediction = Wav2VecPrediction(channels=channels)

        # Calculate offset for prediction module
        # NOT SURE THAT WE NEED THIS?!
        # def calc_offset():
        #     jin = 0
        #     rin = 0
        #     for layer in next(self.encoder.children()):
        #         if layer.__class__.__name__ == 'Conv1d':
        #             k = layer.kernel_size[0]
        #             stride = layer.stride[0]
        #             if rin == 0:
        #                 rin = k
        #             rin = rin + (k - 1) * jin
        #             if jin == 0:
        #                 jin = stride
        #             else:
        #                 jin *= stride
        #     offset = math.ceil(rin / jin)
        #
        #     return int(offset)
        # self.offset = calc_offset()

    def forward(self, x):
        z = self.encoder(x)
        c = self.context(z)

        z, z_n, c = self.prediction(c, z)
        z_n = z_n.squeeze(0)

        channels = c.shape[1]
        length = c.shape[2]
        k = c.shape[3]

        pred_buffer = torch.zeros(3, channels * length * k)

        for i in range(k):
            pred_buffer[0][(length * channels) * i:(length * channels) * (i + 1)] = torch.flatten(input=c[..., :, :, i])

            pred_buffer[1][(length * channels) * i:(length * channels) * (i + 1)] = torch.flatten(input=F.pad(
                input=z[..., i + 1:].transpose(0, 1),
                pad=(i + 1, 0, 0, 0), mode='constant',
                value=0))

            pred_buffer[2][(length * channels) * i:(length * channels) * (i + 1)] = torch.flatten(input=F.pad(
                input=z_n[..., i + 1:].transpose(0, 1),
                pad=(i + 1, 0, 0, 0), mode='constant',
                value=0))

        return pred_buffer[0], pred_buffer[1], pred_buffer[2]


class Encoder(nn.Module):
    def __init__(self, channels, activation, dropout):
        super(Encoder, self).__init__()

        def encoder_conv_block(n_in, n_out, kernel_size, stride, dropout, activation):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride),
                nn.Dropout(p=dropout),
                nn.GroupNorm(1, n_out, affine=True),
                activation
            )

        # (in_dim, out_dim, kernel, stride)
        # .
        # .
        # layer_n
        self.layers = [
            (1, channels, 10, 5),
            (channels, channels, 8, 4),
            (channels, channels, 8, 4),
            (channels, channels, 4, 2),
            (channels, channels, 4, 2),
            (channels, channels, 4, 2)
        ]

        self.conv_blocks = nn.ModuleList()

        for n_in, n_out, kernel_size, stride in self.layers:
            self.conv_blocks.append(encoder_conv_block(n_in, n_out, kernel_size, stride, dropout, activation))

        self.encoder = nn.Sequential(*self.conv_blocks)

    def forward(self, x):
        x = self.encoder(x)
        x = log_compression(x)
        # TODO implement skipped connections?
        return x


class Context(nn.Module):
    def __init__(self, channels, k, dropout, activation, layers=10):
        super(Context, self).__init__()

        # All block are the same, so create using a function
        def context_conv_block(n_in, n_out, k, dropout, activation):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, padding=1),
                nn.Dropout(p=dropout),
                nn.GroupNorm(1, n_out, affine=False),
                activation
            )

        # Holder for conv layers
        self.conv_blocks = nn.ModuleList()

        # Create #layers number of conv-blocks
        for i in range(0, layers):
            self.conv_blocks.append(context_conv_block(channels, channels, k, dropout, activation))

        self.context = nn.Sequential(*self.conv_blocks)

    def forward(self, z):
        c = self.context(z)
        return c


class Wav2VecPrediction(nn.Module):
    def __init__(self, channels, prediction_steps=12):
        super(Wav2VecPrediction, self).__init__()
        self.transpose_context = nn.ConvTranspose2d(channels, channels, (1, prediction_steps))
        self.sample_distance = None
        self.n_negatives = 1

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape

        y = y.transpose(0, 1)  # BCT -> CBT
        y = y.contiguous().view(fsz, -1)  # CBT => C(BxT)

        high = tsz if self.sample_distance is None else min(tsz, self.sample_distance)
        assert high > 1

        neg_idxs = torch.randint(low=0, high=high, size=(bsz, self.n_negatives * tsz))

        with torch.no_grad():
            if self.n_negatives > 0:
                tszs = (
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
        # get distractor samples
        z_n = self.sample_negatives(z)

        return z, z_n, c


class ZeroPad1d(nn.Module):
    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return F.pad(x, (self.pad_left, self.pad_right))


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def plot_wav(waveform):
        plt.figure(1)
        plt.title("Test wave")
        plt.plot(waveform[0])
        plt.show()


    waveform, sample_rate = torchaudio.load("wav_16k_example.wav")
    # torch.unsqueeze(waveform, 1)
    print(waveform.shape)
    # plot_wav(waveform)
    model = Wav2vec()
    # For testing unsqueeze to match conv1d shape requirements
    z, c = model(torch.unsqueeze(waveform, 1))
    print(z)
    # print(waveform)
