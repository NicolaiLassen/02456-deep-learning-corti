import matplotlib.pyplot as plt
import torch
import torchaudio

from models.wav2vec import Wav2vec


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    # log σ(X^T . Y))
    def log_sigmoid_probs(self, x, y):
        # Z^T . HK
        x_t = x.transpose(0, 1)
        # Take the mean probability of x being y
        out = torch.einsum("ijk,jik->jik", x_t, y)
        # E
        out = torch.mean(out)
        out = torch.sigmoid(out)
        return torch.log(out)

    def forward(self, h_k, z, z_n):
        # - (log σ(Z^T . HK)) + λE [log σ(ZN^T . HK)])
        return - (self.log_sigmoid_probs(z, h_k) + self.log_sigmoid_probs(-z_n, h_k))


# Test criterion
if __name__ == '__main__':
    criterion = ContrastiveLoss()
    model = Wav2vec()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    waveform, sample_rate = torchaudio.load("../models/wav_16k_example.wav")

    loss_values = []
    model.train()
    for i in range(1000):
        optimizer.zero_grad()
        c, z, z_n = model(torch.unsqueeze(waveform, 1))
        loss = criterion(c, z, z_n)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        print(loss.item())

        if i % 100 == 0:
            plt.plot(loss_values)
            plt.show()
