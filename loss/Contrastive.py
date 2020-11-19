import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio

from models.wav2vec import Wav2vec, Wav2VecPrediction


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    # log σ(X^T . Y))
    def log_sigmoid_probs(self, x, y):
        # Z^T . HK
        out = torch.dot(x, y)
        out = torch.sigmoid(out)
        out = torch.log(out + 1e-7)
        return out

    def forward(self, h_k, z, z_n):
        # - (log σ(Z . HK)) + λE [log σ(ZN . HK)])
        return - (self.log_sigmoid_probs(z, h_k) + self.log_sigmoid_probs(-z_n, h_k))


if __name__ == '__main__':
    criterion = ContrastiveLoss()
    modelPre = Wav2vec()
    modelPred = Wav2VecPrediction()

    optimizer = torch.optim.Adam(modelPre.parameters(), lr=0.001)
    waveform, sample_rate = torchaudio.load("../models/wav_16k_example.wav")

    loss_values = []

    modelPre.train()
    for i in range(100):
        optimizer.zero_grad()
        z, c = modelPre(torch.unsqueeze(waveform, 1))
        z, z_n, c = modelPred(c, z)
        z_n = z_n.squeeze(0)

        # print("Z: {}".format(z.shape))
        # print("Z_n: {}".format(z_n.shape))
        # print("c: {}".format(c.shape))

        channels = c.shape[1]
        length = c.shape[2]
        k = c.shape[3]

        preds = torch.zeros(3, c.shape[2] * k * channels)

        for i in range(k):
            preds[0][(length * channels) * i:(length * channels) * (i + 1)] = c[..., :, :, i].flatten()
            preds[1][(length * channels) * i:(length * channels) * (i + 1)] = F.pad(input=z[..., i:].transpose(0, 1),
                                                                                    pad=(0, i, 0, 0), mode='constant',
                                                                                    value=1).flatten()
            preds[2][(length * channels) * i:(length * channels) * (i + 1)] = F.pad(input=z_n[..., i:].transpose(0, 1),
                                                                                    pad=(0, i, 0, 0), mode='constant',
                                                                                    value=1).flatten()

        loss = criterion(preds[0], preds[1], preds[2])
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        print(loss.item())

    plt.plot(loss_values)
    plt.show()
