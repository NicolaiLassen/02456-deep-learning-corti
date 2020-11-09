import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import models
import utils.plot_util as plot_util
from utils.audio_preprocessor_util import AudioPreprocessor

train_on_gpu = torch.cuda.is_available()

if __name__ == '__main__':

    #
    preprocessor = AudioPreprocessor()
    preprocessor.load_data()

    f_name = './resources/wav2vec_small.pt'
    transfer_model = models.Wave2VecTransfer(f_name)

    # TODO: all from params
    parser = argparse.ArgumentParser()

    # TODO: wrap in train evaluate

    # hyper
    learning_rate = 0.001
    n_epochs = 500

    # ref: https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
    # hyper
    margin = 1.0
    p = 2
    criterion = nn.TripletMarginLoss(margin=margin, p=p)

    # ref: https://pytorch.org/docs/stable/optim.html
    optimizer = optim.Adam(transfer_model.parameters(), lr=learning_rate)

    if train_on_gpu:
        transfer_model.cuda()

    # train loop for the transfer and new loss
    for epoch in range(n_epochs):

        # Bookkeeping of loss to plot
        train_loss = 0.0

        transfer_model.train()
        ## TODO: setup train_loader for this data type
        for data, target in train_loader:

            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            anchor = transfer_model(data)

            with torch.no_grad():
                pos = models.glove_vectors.wv[target]
                neg = models.glove_vectors.wv[target]

            loss = criterion(anchor, pos, neg)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)

        # Bookkeeping of loss to plot
        valid_loss = 0.0

        transfer_model.eval()

        # plot embed vectors
        plot_util.TSNE_embed_context_plot()

        torch.save(transfer_model.state_dict(), 'models/'.format("insert model name"))
