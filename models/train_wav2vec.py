import argparse

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchaudio
import torchaudio.datasets
import torch.utils.data as data

from loss.Contrastive import ContrastiveLoss
from models.wav2vec import Wav2vec


def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = start_lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

#Download LibriSpeech train/test dataset
train_data = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-100", download=True)
test_data = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)

#Create collate function to pad each audio in order to have the same seq length
def collate(batch):
    data = [torch.Tensor(t[0]).transpose(0,1) for t in batch]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True).contiguous()
    return [data]

#Create batches
batch_size = 8

train_loader = data.DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                collate_fn=collate,
                                shuffle=True)

test_loader = data.DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                collate_fn=collate,
                                shuffle=False)

#Training
train_on_gpu = torch.cuda.is_available()

if __name__ == '__main__':

    # hyperparameters & batches
    num_epochs = 10
    n_epochs_before_save = 2

    model = Wav2vec()
    
    learning_rate = 0.01

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if train_on_gpu:
        model.cuda()

    train_loss = []
    valid_loss = []

    model.train()

    for epoch in range(num_epochs):        
        train_epoch_loss = 0.0

        for waveform in train_loader:

            if train_on_gpu:
                waveform[0] = waveform[0].cuda()

            optimizer.zero_grad()
          
            c, z, z_n = model(waveform[0].transpose(1,2))

            loss = criterion(c, z, z_n)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item() * waveform[0].size(0)

        train_loss.append(train_epoch_loss)

        if epoch > 0 and epoch % n_epochs_before_save == 0:
            adjust_learning_rate(learning_rate, optimizer, epoch)
            plt.plot(train_loss)
            plt.show()