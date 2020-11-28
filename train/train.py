from utils.training import collate
from models.Wav2VecSemantic import Wav2vecSemantic
from torch.utils.data import DataLoader
from typing import List
import matplotlib.pyplot as plt
import torch
import torchaudio
from transformers import ElectraTokenizer, ElectraModel
from criterion.Contrastive import ContrastiveLoss
from criterion.Dist import DistLoss
from torch import optim


def train_model_semantic(wav2vec: Wav2vecSemantic, optimizer: optim, loss: ContrastiveLoss, epochs: int
                            , training_loader: DataLoader, test_loader: DataLoader, tokenizer
                            , electra_model, dist_criterion) -> (Wav2vecSemantic, List):

    wav_model = wav2vec
    con_criterion = loss
    optimizer = optimizer

    losses = []

    for i in range(epochs):

        # Enter training state
        wav_model.train()
        #print(next(iter(training_loader)))

        for waveform, text in training_loader:

            # Zero gradients
            optimizer.zero_grad()

            # Get electra embeddings
            tokens = tokenizer(text, return_tensors="pt", padding=True)
            e = electra_model(**tokens).last_hidden_state
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
                plt.plot(losses)
                plt.show()

        # TODO make some train and test metrics
        # wav_model.eval()

    return wav_model, losses


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

    # Define wav2vec model, optimizer and criterion
    wav_model = Wav2vecSemantic()
    optimizer = torch.optim.Adam(wav_model.parameters(), lr=0.02)
    con_criterion = ContrastiveLoss()

    # Define electra model, loss and tokenizer
    dist_criterion = DistLoss()
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator', return_dict=True)

    model, losses = train_model_semantic(wav2vec=wav_model, optimizer=optimizer, loss=con_criterion, epochs=1,
                         training_loader=train_loader, test_loader=test_loader, tokenizer=tokenizer,
                         electra_model=electra_model, dist_criterion=dist_criterion)

    """
    
    losses = []

    for i in range(1):

        # Enter training state
        wav_model.train()
        # print(next(iter(training_loader)))

        for waveform, text in train_loader:

            optimizer.zero_grad()
            embed_shape = 0

            tokens = tokenizer(text, return_tensors="pt", padding=True)
            e = electra_model(**tokens).last_hidden_state
            # tokens = torch.tensor(tokenizer.encode(text, return_tensors="pt"))
            # e = electra_model(tokens)[0]
            embed_shape = e.shape[1]

            print("Wave shape: {}".format(waveform.shape))
            # print(e.shape)
            print("Electra embedding shape {}".format(e.shape))
            (hk, z, z_n), e_c = wav_model(x=waveform, idx_n=embed_shape)

            # Calculate contrastive loss / and dist if text data
            loss_con = con_criterion(hk, z, z_n)
            loss_dist = dist_criterion(e, e_c)
            loss = (loss_dist + loss_con) / 2

            losses.append(loss.item())

            # Backprop
            loss.backward()
            optimizer.step()

            if i > 0 and i % 100 == 0:
                plt.plot(losses)
                plt.show()
            #break
            
*************************************************************
    
    train_model_semantic(wav2vec=wav_model, optimizer=optimizer, loss=con_criterion, epochs=1
                            , training_loader=train_loader, test_loader=test_loader,
                            tokenizer=tokenizer, electra_model=electra_model, loss_dist=con_criterion)

    



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
            
            
            
"""

