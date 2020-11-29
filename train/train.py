import pickle
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
from utils.plot_util import TSNE_Wav2Vec_embed_Semantic_embed
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

    epoch_mean_losses = []
    for epoch_i in range(epochs):

        # Enter training state
        wav_model.train()
        epoch_sub_losses = []
        for batch_i, (waveform, text) in enumerate(training_loader):

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

            epoch_sub_losses.append(loss.item())

            X = torch.stack([e_c.view(32, -1), e.view(32, -1)]).view(32 * 2, -1).detach().cpu().numpy()
            TSNE_Wav2Vec_embed_Semantic_embed(X, batch_n=32,
                                              file_name="./TSNE_256_e_{}_b_{}.png".format(epoch_i, batch_i))

            # Backprop
            loss.backward()
            optimizer.step()

        with open('epoch_sub_losses_e_{}.pkl'.format(epoch_i), 'wb') as handle:
            pickle.dump(epoch_sub_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

        torch.save(wav_model.state_dict(), "./semantic_256_e_{}.ckpt".format(epoch_i))
        plt.plot(epoch_sub_losses)
        plt.savefig("./semantic_256_e_{}.ckpt.png".format(epoch_i))

        epoch_mean_losses.append(torch.tensor(epoch_sub_losses).mean().item())

        with open('epoch_mean_losses_e_{}.pkl'.format(epoch_i), 'wb') as handle:
            pickle.dump(epoch_mean_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # TODO make some train and test metrics
        # wav_model.eval()

    return wav_model, epoch_mean_losses


if __name__ == "__main__":
    train_data = torchaudio.datasets.LIBRISPEECH("../data/", url="train-clean-100", download=True)
    test_data = torchaudio.datasets.LIBRISPEECH("../data/", url="test-clean", download=True)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=32,  # 256
                              collate_fn=collate,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_data,
                             batch_size=32,  # 256
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

    model, losses = train_model_semantic(wav2vec=wav_model, optimizer=optimizer, loss=con_criterion, epochs=10,
                                         training_loader=train_loader, test_loader=test_loader, tokenizer=tokenizer,
                                         electra_model=electra_model, dist_criterion=dist_criterion)
