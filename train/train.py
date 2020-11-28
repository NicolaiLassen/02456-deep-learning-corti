from utils.training import collate
from models.wav2vecSemantic import Wav2vecSemantic
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torchaudio
from transformers import ElectraTokenizer, ElectraModel

from criterion.Contrastive import ContrastiveLoss
from criterion.Dist import DistLoss

if __name__ == "__main__":

    train_data = torchaudio.datasets.LIBRISPEECH("../data/", url="train-clean-100", download=True)
    test_data = torchaudio.datasets.LIBRISPEECH("../data/", url="test-clean", download=True)

    train_loader = DataLoader(dataset=train_data,
                                   batch_size=1,
                                   collate_fn=collate,
                                   shuffle=True)

    test_loader = DataLoader(dataset=test_data,
                                  batch_size=1,
                                  collate_fn=collate,
                                  shuffle=False)

    wav_model = Wav2vecSemantic()

    con_criterion = ContrastiveLoss()
    dist_criterion = DistLoss()

    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator', return_dict=True)

    text_1 = "SHE HAD THIN AWKWARD FIGURE".lower()
    waveform, sample_rate = torchaudio.load("./wav_16k_example.wav")
    optimizer = torch.optim.Adam(wav_model.parameters(), lr=0.02)

    loss_dist_values = []
    wav_model.train()
    for i in range(1):

        optimizer.zero_grad()
        tokens = torch.tensor(tokenizer.encode(text_1, return_tensors="pt"))
        e = electra_model(tokens)[0]
        print(e.shape)
        print(tokens.shape)
        (hk, z, z_n), e_c = wav_model(torch.unsqueeze(waveform, 1), tokens.shape[1])

        loss_dist = dist_criterion(e, e_c)
        loss_con = con_criterion(hk, z, z_n)
        loss = (loss_dist + loss_con) / 2

        loss_dist_values.append(loss.item())
        loss.backward()

        optimizer.step()
        print(loss_dist.item())

        if i > 0 and i % 100 == 0:
            plt.plot(loss_dist_values)
            plt.show()


