import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wave2vec_transfer

train_on_gpu = torch.cuda.is_available()

if __name__ == '__main__':

    f_name = '/path/to/wav2vec.pt'
    transfer_model = wave2vec_transfer.Wave2VecTransfer(f_name)

    # TODO: Set from params
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

    # Bookkeeping
    valid_loss_min = np.Inf

    if train_on_gpu:
        transfer_model.cuda()

    # train loop for the transfer and new loss
    for epoch in range(n_epochs):
        transfer_model.train()

        transfer_model.eval()
