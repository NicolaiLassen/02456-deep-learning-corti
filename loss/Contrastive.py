import torch
import torchaudio

from models.wav2vec import Wav2vec, Wav2VecPrediction


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    # log σ(X^T . Y))
    def log_sigmoid_probs(self, x, y):
        x_t = torch.transpose(x, 0, 1)
        # Z^T . HK
        out = torch.einsum("jikt,ijkt->i", x_t, y)
        print(out)
        out = torch.sigmoid(out)
        print(out)
        out = torch.log(out)
        print(out)
        return out

    def forward(self, h_k, z, z_n):
        # - (log σ(Z . HK)) + λE [log σ(ZN . HK)])
        return - (self.log_sigmoid_probs(z, h_k) + self.log_sigmoid_probs(-z_n, h_k))


if __name__ == '__main__':
    criterion = ContrastiveLoss()
    modelPre = Wav2vec()
    modelPred = Wav2VecPrediction()

    optimizer = torch.optim.Adam(modelPre.parameters(), lr=0.0001)
    waveform, sample_rate = torchaudio.load("../models/wav_16k_example.wav")

    for i in range(1):
        optimizer.zero_grad()
        z, c = modelPre(torch.unsqueeze(waveform, 1))
        z, z_n, c = modelPred(c, z)
        z = z.unsqueeze(-1)

        # ### NOT SURE about this
        z_n = z_n.permute([0, 2, 3, 1])
        z = z.repeat(1, 1, 1, c.shape[3]) # <--- add the same z for all steps of C in the future
        z_n = z_n.repeat(1, 1, 1, c.shape[3]) # <--- add the same z_n for all steps of C in the future
        #####################################

        loss = criterion(c, z, z_n)

        loss.backward()
        optimizer.step()

        # print("loss", loss)
