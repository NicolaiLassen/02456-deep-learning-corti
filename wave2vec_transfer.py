from abc import ABC

import torch
import torch.nn as nn

from fairseq.fairseq.models.wav2vec import Wav2VecModel


class Wave2VecTransfer(nn.Module):
    """ Wrapper model for transfer learning """

    # transfer learning:
    # ref: https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce

    def __init__(self, f_name):
        super().__init__()

        # step 1 inject wave2vec
        wave2vec = PretrainedWav2VecModel(f_name)

        #

    def forward(self, input):

        # Get embeds latent space from PretrainedWav2VecModel

        out = input
        return out


# ref: https://github.com/mailong25/vietnamese-speech-recognition/blob/master/wav2vec.py
class PretrainedWav2VecModel(nn.Module):
    def __init__(self, f_name):
        super().__init__()

        # TODO: CPU?
        checkpoint = torch.load(f_name, map_location=torch.device('cpu'))
        self.args = checkpoint["args"]
        model = Wav2VecModel.build_model(self.args, None)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        self.model = model

    def forward(self, x):
        with torch.no_grad():
            z = self.model.feature_extractor(x)
            if isinstance(z, tuple):
                z = z[0]
            c = self.model.feature_aggregator(z)
        return z, c
