import torch
import torch.nn as nn

from fairseq.fairseq.models.wav2vec import Wav2VecModel


class Wave2VecTransfer(nn.Module):
    """
    Wrapper model for transfer learning.
    :param f_name denotes the location of the pretraind wave2ve cp
    """

    # transfer learning:
    # ref: https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce

    def __init__(self, f_name):
        super().__init__()

        # step 1 inject wave2vec
        # should we
        self.wave2vec = PretrainedWav2VecModel(f_name)

        # step 2 harvest embed latent dim
        self.embed_z = nn.Embedding()  # TODO: use latent dim out of wave2vec
        self.embed_c = nn.Embedding()  # TODO:

    def forward(self, input):
        # Get embeds latent space from PretrainedWav2VecModel
        out = self.wave2vec(input)

        # TODO: train via bp on our latent dim of the model

        # embed the z dim in our latent -> transformer
        out = self.embed_z(out)
        # embed the c dim in our latent -> transformer
        out = self.embed_c(out)

        return out


class Vec2Word(nn.Module):
    """
    Extracts features of the embed space to
    """

    # TODO: we should convert the embeds to speech
    def __init__(self):
        super().__init__()


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
