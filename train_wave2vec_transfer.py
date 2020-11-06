import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wave2vec_transfer

train_on_gpu = torch.cuda.is_available()

if __name__ == '__main__':

    f_name = './pretrained_cp/wav2vec_small.pt'
    transfer_model = wave2vec_transfer.Wave2VecTransfer(f_name)

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

        # Bookkeeping
        test_loss = np.Inf
        valid_loss = np.Inf

        transfer_model.train()

        transfer_model.eval()

        torch.save(transfer_model.state_dict(), 'models/'.format("insert model name"))
