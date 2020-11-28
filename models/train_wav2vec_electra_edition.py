import argparse

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from criterion.Contrastive import ContrastiveLoss
from models.Wav2Vec import Wav2vec

import torchaudio.datasets

import torch.utils.data as data


train_on_gpu = torch.cuda.is_available()

if __name__ == '__main__':

    # hyperparameters & batches
    num_epochs = 1

    model = Wav2vec()

    learning_rate = 0.01

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # if train_on_gpu:
    #    model.cuda()

    train_loss = []
    valid_loss = []

    model.train()

    for epoch in range(num_epochs):

        data_len = len(train_loader.dataset)

        train_epoch_loss = 0.0

        for batch_idx, waveform in enumerate(train_loader):
            print(waveform[0].shape)

            optimizer.zero_grad()

            c, z, z_n = model(waveform[0].transpose(1, 2))

            loss = criterion(c, z, z_n)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item() * waveform[0].size(0)

            train_loss.append(train_epoch_loss)

            if batch_idx % 10 == 0 or batch_idx == data_len:
                print("Train Epoch %2i ({:5.2f}%) : Train Loss %f" % (  # \t Test criterion: %f
                    epoch, epoch / num_epochs * 100, loss.item()))  # , test_loss[-1]

    plt.plot(train_loss)
    plt.show()
    # plt.plot(test_loss);