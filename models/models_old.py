import gensim.downloader
import torch
import torch.nn as nn

from fairseq.fairseq.models.wav2vec import Wav2VecModel

# gensim word2vec embeds
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')


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
        self.embed_z = nn.Embedding(2, 2)  # TODO: use latent dim out of wave2vec
        self.embed_c = nn.Embedding(2, 2)  # TODO: same ^

        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, input):
        # Get embeds latent space from PretrainedWav2VecModel
        z, c = self.wave2vec(input)

        # TODO: train via bp on our latent dim of the model

        # embed the z dim in our latent -> transformer
        # z = self.embed_z(out)
        # embed the c dim in our latent -> transformer
        # c = self.embed_c(out)

        return self.softmax_out(), z, c


class Vec2Semantic(nn.Module):

    def __init__(self):
        super().__init__()

        # ref: https://huggingface.co/bert-base-uncased?
        # step 1: use BERT embbedings
        # self.bert_embed =

        #
        # self.rnn =


# ref: https://static.googleusercontent.com/media/research.google.com/da//pubs/archive/42543.pdf
class Vec2Word(nn.Module):
    """
    Extracts features of the embed space and uses the labels to transform them into text
    """

    # TODO: we should convert the embeds to speech

    def __init__(self):
        super().__init__()

        # step 1:
        self.embed_c = nn.Embedding()

        # step 2: Use recurrent to prevent lossing the grad and keep the idea of a previous word
        # self.rnn =

    # ref: https://github.com/mailong25/vietnamese-speech-recognition/blob/master/wav2vec.py


class PretrainedWav2VecModel(nn.Module):
    '''
    Learns speech representations on unlabeled data
    '''

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
