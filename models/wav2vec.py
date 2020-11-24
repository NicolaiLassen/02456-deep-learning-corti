import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self,
                 channels=512,
                 activation=nn.ReLU(),
                 dropout=0.1
                 ):
        super(Wav2vec, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        encoder_layers = [
            (1, channels, 10, 5),
            (channels, channels, 8, 4),
            (channels, channels, 4, 2),
            (channels, channels, 4, 2),
            (channels, channels, 4, 2)
        ]

        context_layers = 10

        self.encoder = Encoder(channels=channels,
                               activation=activation,
                               dropout=dropout,
                               layers=encoder_layers
                               )
        self.context = Context(channels=channels,
                               kernel_size=3,
                               activation=activation,
                               dropout=dropout,
                               layers=context_layers
                               )
        self.prediction = Wav2VecPrediction(channels=channels)

    def forward(self, x):
        z = self.encoder(x)
        c = self.context(z)
        hk, z, z_n = self.prediction(c, z)
        z_n = z_n.squeeze(0)

        channels = hk.shape[1]
        length = hk.shape[2]

        # sum_k=1^K
        k_start = 1
        prediction_steps = hk.shape[3] - k_start

        prediction_buffer = torch.zeros(channels * length * prediction_steps)
        target_buffer = torch.zeros(channels * length * prediction_steps)
        target_n_buffer = torch.zeros(channels * length * prediction_steps)

        # sum_k=1^K
        # TODO clean this method to be more optim! Verbose starting point
        # We only need this for the Z
        for i in range(k_start, prediction_steps):
            prediction_buffer[(length * channels) * i:(length * channels) * (i + 1)] = torch.flatten(
                input=hk[..., :, :, i])

            target_buffer[(length * channels) * i:(length * channels) * (i + 1)] = torch.flatten(F.pad(
                input=z[..., i + 1:],
                pad=(i + 1, 0, 0, 0), mode='constant',
                value=0))

            target_n_buffer[(length * channels) * i:(length * channels) * (i + 1)] = torch.flatten(F.pad(
                input=z_n[..., i + 1:],
                pad=(i + 1, 0, 0, 0), mode='constant',
                value=0))

        return prediction_buffer.view(channels, length, prediction_steps), \
               target_buffer.view(channels, length, prediction_steps), \
               target_n_buffer.view(channels, length, prediction_steps)


class Encoder(nn.Module):
    def __init__(self, channels, activation, dropout, layers):
        super(Encoder, self).__init__()

        def encoder_conv_block(n_in, n_out, kernel_size, stride, dropout, activation):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride),
                nn.Dropout(p=dropout),
                nn.GroupNorm(1, n_out, affine=True),
                activation
            )

        self.conv_blocks = nn.ModuleList()

        for n_in, n_out, kernel_size, stride in layers:
            self.conv_blocks.append(encoder_conv_block(n_in, n_out, kernel_size, stride, dropout, activation))

        self.encoder = nn.Sequential(*self.conv_blocks)

    def forward(self, x):
        x = self.encoder(x)
        x = log_compression(x)
        return x


class Context(nn.Module):
    def __init__(self, channels, kernel_size, dropout, activation, layers):
        super(Context, self).__init__()

        def context_conv_block(n_in, n_out, kernel_size, dropout, activation):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, kernel_size, padding=1),
                nn.Dropout(p=dropout),
                nn.GroupNorm(1, n_out, affine=False),
                activation
            )

        self.conv_blocks = nn.ModuleList()

        for i in range(0, layers):
            self.conv_blocks.append(context_conv_block(channels, channels, kernel_size, dropout, activation))

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

    # lambda_n = 1
    # https://github.com/pytorch/fairseq/blob/master/fairseq/models/wav2vec/wav2vec.py
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
        # Transpose to give steps predictions into the future
        c = self.transpose_context(c.unsqueeze(-1))
        # get distractor samples
        z_n = self.sample_negatives(z)
        return c, z, z_n
