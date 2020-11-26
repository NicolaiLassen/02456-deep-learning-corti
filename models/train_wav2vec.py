import argparse

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchaudio

from loss.Contrastive import ContrastiveLoss
from models.wav2vec import Wav2vec

train_on_gpu = torch.cuda.is_available()


def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = start_lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    # hyper
    n_epochs_before_save = 200
    n_epochs = 1000
    n_batches = 256
    learning_rate = 0.01

    # load waves
    # preprocessor = AudioPreprocessor()
    # preprocessor.load_data()
    # TODO: use AudioPreprocessor not just torchaudio loader
    waveform, sample_rate = torchaudio.load("../models/wav_16k_example.wav")

    model = Wav2vec()
    # print(model)

    # TODO: all from params
    parser = argparse.ArgumentParser()

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if train_on_gpu:
        model.cuda()

    train_loss = []

    # train loop for the transfer and new loss
    for epoch in range(n_epochs):

        # Bookkeeping of loss to plot
        train_epoch_loss = 0.0

        model.train()

        # TODO: setup train_loader for this data type
        for data in [waveform]:

            if train_on_gpu:
                data = data.cuda()

            optimizer.zero_grad()
            c, z, z_n = model(torch.unsqueeze(data, 1))
            loss = criterion(c, z, z_n)
            loss.backward()
            optimizer.step()

            # TODO: remove this when all is ready
            print(loss.item())
            train_epoch_loss += loss.item() * data.size(0)

        train_loss.append(train_epoch_loss)

        if epoch > 0 and epoch % n_epochs_before_save == 0:
            adjust_learning_rate(learning_rate, optimizer, epoch)
            # TODO: plot loss here
            plt.plot(train_loss)
            plt.show()

        # TODO eval the model here:
        # Bookkeeping of loss to plot
        # valid_loss = 0.0
        # model.eval()

        # plot embed vectors
        # plot_util.TSNE_embed_context_plot()
