import torch
import torchaudio
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import ElectraTokenizer, ElectraModel

from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate

if __name__ == '__main__':

    batch_size = 4
    wav_model = Wav2vecSemantic(channels=256, prediction_steps=6)
    train_data = torchaudio.datasets.LIBRISPEECH("./data/", url="train-clean-100", download=False)
    training_loader = DataLoader(dataset=train_data,
                                 batch_size=batch_size,
                                 pin_memory=True,
                                 collate_fn=collate,
                                 shuffle=False)

    triplet_criterion = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=torch.nn.PairwiseDistance())

    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator', return_dict=True)

    nhead = 4
    head_dim = 64
    d_model = nhead * head_dim
    # ref: https://arxiv.org/pdf/1706.03762.pdf
    net = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=1, num_decoder_layers=1)
    optimizer = Adam(net.parameters(), lr=0.000001)

    for wave, texts_p in training_loader:
        optimizer.zero_grad()

        c, z = wav_model(wave)
        c1 = nn.Conv1d(256, 256, kernel_size=20, stride=20, padding=1)

        z = c1(z)

        z_in = z.permute(2, 0, 1)

        (_, texts_n) = next(iter(training_loader))

        texts_n = texts_n[:batch_size]
        texts_p = texts_p[:batch_size]

        tokens = tokenizer([*texts_p, *texts_n], return_tensors="pt", padding=True)
        e_embed = electra_model(**tokens).last_hidden_state
        e_embed_in = e_embed[:batch_size].permute(1, 0, 2)

        out = net(z_in, e_embed_in)
        loss = triplet_criterion(out, e_embed_in, e_embed[batch_size:batch_size * 2].permute(1, 0, 2))

        loss.backward()
        optimizer.step()
        print(loss)
