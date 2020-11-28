from typing import List

import matplotlib.pyplot as plt
import torch
import torchaudio
from torch import optim
from torch.utils.data import DataLoader
from transformers import ElectraTokenizer, ElectraModel

from criterion.Contrastive import ContrastiveLoss
from criterion.Dist import DistLoss
from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate

# TODO: MORE GPU !!
train_on_gpu = False  # torch.cuda.is_available()


def train_model_semantic(wav2vec: Wav2vecSemantic, optimizer: optim, loss: ContrastiveLoss, epochs: int
                         , training_loader: DataLoader, test_loader: DataLoader, tokenizer
                         , electra_model, dist_criterion) -> (Wav2vecSemantic, List):
    wav_model = wav2vec

    if train_on_gpu:
        wav_model.cuda()

    con_criterion = loss
    optimizer = optimizer

    losses = []

    for i in range(epochs):

        # Enter training state
        wav_model.train()
        # print(next(iter(training_loader)))

        for waveform, text in training_loader:

            if train_on_gpu:
                waveform = waveform.cuda()

            # Zero gradients
            optimizer.zero_grad()

            # Get electra embeddings
            tokens = tokenizer(text, return_tensors="pt", padding=True)

            e = electra_model(**tokens).last_hidden_state

            if train_on_gpu:
                e = e.cuda()

            embed_shape = e.shape[1]

            # Forward pass through architecture
            (hk, z, z_n), e_c = wav_model(x=waveform, idx_n=embed_shape)

            # Calculate contrastive loss / and dist if text data
            loss_con = con_criterion(hk, z, z_n)
            loss_dist = dist_criterion(e, e_c)
            loss = (loss_dist + loss_con) / 2
            print(loss)

            losses.append(loss.item())

            # Backprop
            loss.backward()
            optimizer.step()

            if i > 0 and i % 100 == 0:
                torch.save(model.state_dict(), "./semantic_256_e_{}.ckpt".format(i))
                plt.plot(losses)
                plt.savefig("./semantic_256_e_{}.ckpt.png".format(i))

        # TODO make some train and test metrics
        # wav_model.eval()

    return wav_model, losses


if __name__ == "__main__":
    train_data = torchaudio.datasets.LIBRISPEECH("../data/", url="train-clean-100", download=True)
    test_data = torchaudio.datasets.LIBRISPEECH("../data/", url="test-clean", download=True)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=40,  # 128
                              collate_fn=collate,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_data,
                             batch_size=40,  # 128
                             collate_fn=collate,
                             shuffle=False)

    # Define wav2vec model, optimizer and criterion
    wav_model = Wav2vecSemantic(channels=256)
    optimizer = torch.optim.Adam(wav_model.parameters(), lr=0.001)
    con_criterion = ContrastiveLoss()

    # Define electra model, loss and tokenizer
    dist_criterion = DistLoss()
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator', return_dict=True)

    model, losses = train_model_semantic(wav2vec=wav_model, optimizer=optimizer, loss=con_criterion, epochs=1,
                                         training_loader=train_loader, test_loader=test_loader, tokenizer=tokenizer,
                                         electra_model=electra_model, dist_criterion=dist_criterion)
