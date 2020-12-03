import pickle
import matplotlib.pyplot as plt
import torch
from models.Wav2VecSemantic import Wav2vecSemantic
from models.Wav2Vec import Wav2vec
import torchaudio
from torch.utils.data import DataLoader
from utils.training import collate
from criterion.CosineDist import CosineDistLoss
import numpy as np


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

if __name__ == "__main__":

    # Create models and load their weights - basic wav2vec and with text data
    wav_base = Wav2vecSemantic(channels=256)
    wav_base.load_state_dict(torch.load("./ckpt_base_wav2vec/model/wav2vec_semantic_256_e_9.ckpt", map_location=torch.device('cpu')))
    wav_base.eval()

    wav_semantic = Wav2vecSemantic(channels=256)
    wav_semantic.load_state_dict(torch.load("./ckpt/model/wav2vec_semantic_256_e_9.ckpt", map_location=torch.device('cpu')))
    wav_semantic.eval()

    # Dataset for testing
    """
    test_data = torchaudio.datasets.LIBRISPEECH("./data/", url="test-clean", download=True)
    batch_size = 1
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             pin_memory=True,
                             collate_fn=collate,
                             shuffle=False)
    """

    train_data = torchaudio.datasets.LIBRISPEECH("./data/", url="train-clean-100", download=True)
    batch_size = 1
    train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  collate_fn=collate,
                                  shuffle=True)
    # Loss comparison
    loss = CosineDistLoss()

    for name, param in wav_base.named_parameters():
        if name == "prediction.transpose_context.weight":
            print(name, param.data)

