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
        out = x * y
        out = torch.sigmoid(out)
        return torch.log(out).mean()

    def forward(self, h_k, z, z_n):
        # - (log σ(Z^T . HK)) + λE [log σ(ZN^T . HK)])
        return - (self.log_sigmoid_probs(z, h_k) + self.log_sigmoid_probs(-z_n, h_k))


if __name__ == '__main__':
    criterion = ContrastiveLoss()
    model = Wav2vec()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    waveform, sample_rate = torchaudio.load("../models/wav_16k_example.wav")

    loss_values = []

    model.train()
    for i in range(100):
        optimizer.zero_grad()
        c, z, z_n = model(torch.unsqueeze(waveform, 1))

        loss = criterion(z, z_n, c)
        print(loss.item())
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

    plt.plot(loss_values)
    plt.show()
