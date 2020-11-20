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
        out = torch.dot(x, y)
        print(out)
        out = torch.sigmoid(out)
        out = torch.log(out)
        return out

    def forward(self, h_k, z, z_n):
        # - (log σ(Z . HK)) + λE [log σ(ZN . HK)])
        return - (self.log_sigmoid_probs(z, h_k) + self.log_sigmoid_probs(-z_n, h_k))


if __name__ == '__main__':
    criterion = ContrastiveLoss()
    model = Wav2vec()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    waveform, sample_rate = torchaudio.load("../models/wav_16k_example.wav")

    loss_values = []

    model.train()
    for i in range(10):
        optimizer.zero_grad()
        z, z_n, c = model(torch.unsqueeze(waveform, 1))

        loss = criterion(z, z_n, c)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        print(loss.item())

    plt.plot(loss_values)
    plt.show()
