import numpy as np


import torch.nn as nn
import torch.optim as optim
import torch

import wave2vec_transfer

train_on_gpu = torch.cuda.is_available()

if __name__ == '__main__':
    transfer_model = wave2vec_transfer.Wave2VecTransfer()

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(transfer_model.parameters(), lr=0.003)
    n_epochs = 500
    valid_loss_min = np.Inf

    if train_on_gpu:
        transfer_model.cuda()

    for epoch in range(n_epochs):

        transfer_model.train()



