import argparse

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchaudio

from loss.Contrastive import ContrastiveLoss
from models.wav2vec import Wav2vec
from utils.audio_preprocessor_util import AudioPreprocessor

train_on_gpu = torch.cuda.is_available()

if __name__ == '__main__':

    # load waves
    preprocessor = AudioPreprocessor()
    preprocessor.load_data()
    # TODO: use AudioPreprocessor not just torchaudio loader
    waveform, sample_rate = torchaudio.load("../models/wav_16k_example.wav")

    model = Wav2vec()

    # TODO: all from params
    parser = argparse.ArgumentParser()

    # hyper
    learning_rate = 0.01
    n_epochs = 100

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

        ## TODO: setup train_loader for this data type
        for data in [waveform]:

            if train_on_gpu:
                data, target = data.cuda()

            optimizer.zero_grad()
            c, z, z_n = model(torch.unsqueeze(waveform, 1))
            loss = criterion(c, z, z_n)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item() * data.size(0)

        train_loss.append(train_epoch_loss)

        if epoch % 100 == 0:
            plt.plot(train_loss)
            plt.show()

        # Bookkeeping of loss to plot
        # valid_loss = 0.0
        # model.eval()

        # plot embed vectors
        # plot_util.TSNE_embed_context_plot()
