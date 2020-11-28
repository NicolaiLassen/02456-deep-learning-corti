from utils.training import collate
import torchaudio
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from criterion.Contrastive import ContrastiveLoss
from models.wav2vec import Wav2vec

if __name__ == "__main__":

    train_data = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-100", download=True)
    test_data = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)

    train_loader = DataLoader(dataset=train_data,
                                   batch_size=10,
                                   collate_fn=collate,
                                   shuffle=True)

    test_loader = DataLoader(dataset=test_data,
                                  batch_size=10,
                                  collate_fn=collate,
                                  shuffle=False)