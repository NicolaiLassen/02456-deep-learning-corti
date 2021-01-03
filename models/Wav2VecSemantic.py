import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def log_compression(x: Tensor) -> Tensor:
    # https://www.edn.com/log-1-x-compression/
    x = x.abs()
    x = x + 1
    return x.log()


class Wav2vecSemantic(nn.Module):

    def __init__(self,
                 channels=256,
                 activation=nn.ReLU(),
                 dropout=0.1,
                 prediction_steps=12
                 ):
        super(Wav2vecSemantic, self).__init__()

        encoder_layers = [
            (1, channels, 10, 5),
            (channels, channels, 8, 4),
            (channels, channels, 4, 2),
        ]

        context_layers = 5

        self.encoder = Encoder(activation=activation,
                               dropout=dropout,
                               layers=encoder_layers
                               )
        self.context = Context(channels=channels,
                               kernel_size=3,
                               activation=activation,
                               dropout=dropout,
                               layers=context_layers
                               )
        self.prediction = Wav2VecPrediction(channels=channels, prediction_steps=prediction_steps)

        self.feature_out_size = channels

        # ref: https://arxiv.org/pdf/1706.03762.pdf
        nhead = 2
        head_dim = 128
        d_model = nhead * head_dim
        self.down_sample = nn.Conv1d(channels, channels, kernel_size=20, stride=20, padding=1)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=2, num_decoder_layers=2)

    def feature_transform(self, z, context):
        # f: Z -> Q
        z = self.down_sample(z)
        z = z.permute(2, 0, 1)
        return self.transformer(z, context)

    def forward(self, x, contrastive_train=False, context=None):
        z = self.encoder(x)  # f: X -> Z
        c = self.context(z)  # f: Z -> C

        # Case eval
        if not contrastive_train and context is None:
            return c, z

        # Case: train only embed
        if not contrastive_train:
            return self.feature_transform(z, context)

        # Case: enable contrastive learnings
        if contrastive_train:

            # k steps into future
            hk, z, z_n = self.prediction(c, z)

            # shapes
            z_n = z_n.squeeze(0)
            batch = hk.shape[0]
            channels = hk.shape[1]
            length = hk.shape[2]

            # sum_k=1^K
            k_start = 1
            # calc buffer size
            prediction_steps = hk.shape[3] - k_start
            pred_step_range = batch * channels * length
            pred_step_batch_range = pred_step_range * prediction_steps

            # buffers
            prediction_buffer = torch.zeros(pred_step_batch_range)
            target_buffer = torch.zeros(pred_step_batch_range)
            target_n_buffer = torch.zeros(pred_step_batch_range)

            # create prediction step vectors
            for i in range(k_start, prediction_steps):
                prediction_buffer[pred_step_range * i:pred_step_range * (i + 1)] = torch.flatten(
                    input=hk[..., :, :, i])

                target_buffer[pred_step_range * i:pred_step_range * (i + 1)] = torch.flatten(F.pad(
                    input=z[..., i + 1:],
                    pad=(i + 1, 0, 0, 0),
                    mode='constant',
                    value=0))

                target_n_buffer[pred_step_range * i:pred_step_range * (i + 1)] = torch.flatten(F.pad(
                    input=z_n[..., i + 1:],
                    pad=(i + 1, 0, 0, 0),
                    mode='constant',
                    value=0))

            contrastive_pred = (
                prediction_buffer.view(batch, channels, length, prediction_steps),
                target_buffer.view(batch, channels, length, prediction_steps),
                target_n_buffer.view(batch, channels, length, prediction_steps)
            )

            # Case: train only contrastive
            if context is None:
                return contrastive_pred

            # Case: train mixed contrastive and supervised
            return contrastive_pred, self.feature_transform(z, context)


class Encoder(nn.Module):
    def __init__(self, activation, dropout, layers):
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
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        # Context layers
        # Layer 1
        self.c1 = nn.Conv1d(channels, channels, kernel_size, padding=1)
        self.norm1 = nn.GroupNorm(1, channels, affine=True)
        # Layer 2
        self.c2 = nn.Conv1d(channels, channels, kernel_size, padding=1)
        self.norm2 = nn.GroupNorm(1, channels, affine=True)
        # Layer 3
        self.c3 = nn.Conv1d(channels, channels, kernel_size, padding=1)
        self.norm3 = nn.GroupNorm(1, channels, affine=True)
        # Layer 4
        self.c4 = nn.Conv1d(channels, channels, kernel_size, padding=1)
        self.norm4 = nn.GroupNorm(1, channels, affine=True)
        # Layer 5
        self.c5 = nn.Conv1d(channels, channels, kernel_size, padding=1)
        self.norm5 = nn.GroupNorm(1, channels, affine=True)

    def forward(self, z):
        # Layer 1
        residual = z
        c = self.c1(z)
        c = self.dropout(c)
        c = self.norm1(c)
        c = self.activation(c)
        c += residual
        residual = c

        # Layer 2
        c = self.c2(c)
        c = self.dropout(c)
        c = self.norm2(c)
        c = self.activation(c)
        c += residual
        residual = c

        # Layer 3
        c = self.c3(c)
        c = self.dropout(c)
        c = self.norm3(c)
        c = self.activation(c)
        c += residual
        residual = c

        # Layer 4
        c = self.c4(c)
        c = self.dropout(c)
        c = self.norm4(c)
        c = self.activation(c)
        c += residual
        residual = c

        # Layer 5
        c = self.c5(c)
        c = self.dropout(c)
        c = self.norm5(c)
        c = self.activation(c)
        c += residual
        return c


class Wav2VecPrediction(nn.Module):
    def __init__(self, channels, prediction_steps=12):
        super(Wav2VecPrediction, self).__init__()
        self.transpose_context = nn.ConvTranspose2d(channels, channels, (1, prediction_steps))
        self.sample_distance = None
        self.n_negatives = 1

    # lambda_n = 1
    # negative sample function from fairseq
    # ref: https://github.com/pytorch/fairseq/blob/master/fairseq/models/wav2vec/wav2vec.py
    def sample_negatives(self, y: Tensor) -> Tensor:
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
