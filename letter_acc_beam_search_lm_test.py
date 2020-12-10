import math
import os

import numpy as np
import seaborn as sns
import torch
import torchaudio
from torch.utils.data import DataLoader

from decode.src import BeamSearch
from models.Wav2LetterEmbed import Wav2LetterEmbed
from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate


sns.set()

train_on_gpu = torch.cuda.is_available()


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":

    classes = '-abcdefghijklmnopqrstuvwxyz'

    char2index = {
        "-": 0,
        **{chr(i + 96): i for i in range(1, 27)},
        " ": 27
    }

    print(char2index)
    index2char = {
        0: "-",
        **{i + 1: chr(i + 96) for i in range(1, 27)},
        27: " "
    }

    num_features = 256
    wav2letter = Wav2LetterEmbed(num_classes=len(char2index), num_features=num_features)

    wav_base = Wav2vecSemantic(channels=256, prediction_steps=6)
    wav_base.load_state_dict(
        torch.load("./ckpt_con/model/wav2vec_semantic_con_256_e_30.ckpt",
                   map_location=torch.device('cpu')))

    test_data = torchaudio.datasets.LIBRISPEECH("./data/", url="test-clean", download=True)
    batch_size = 1
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             pin_memory=True,
                             collate_fn=collate,
                             shuffle=False)

    create_dir("./ckpt_acc_wav")
    create_dir("./ckpt_acc_wav/losses")
    create_dir("./ckpt_acc_wav/model")

    if train_on_gpu:
        wav_base.cuda()
        wav2letter.cuda()

    epochs = 10000

    ctc_loss = torch.nn.CTCLoss()
    optimizer = torch.optim.Adam(wav2letter.parameters(), lr=1e-5)

    wav_base.eval()
    wav2letter.train()


    def get_y_idxs(texts):
        with torch.no_grad():
            y = []
            len_max = len(max(texts, key=len))
            for i, sentence in enumerate(texts):
                y.append([])
                for char in sentence.lower():
                    try:
                        y[i].append(char2index[char])
                    except:
                        continue
                y[i] += [1] * (len_max - len(y[i]))
            return torch.tensor(y)


    wave, text = next(iter(test_loader))

    for epoch_i in range(epochs):
        epoch_sub_losses = []
        # for batch_i, (wave, texts) in enumerate(test_loader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        y_t = get_y_idxs(text)

        if train_on_gpu:
            y_t = y_t.cuda()
            wave = wave.cuda()

        with torch.no_grad():
            _, c = wav_base(wave)

        a = wav2letter(c)

        a_t = a.transpose(1, 2).transpose(0, 1)
        input_lengths = torch.full((batch_size,), a_t.size()[0], dtype=torch.long)
        target_lengths = torch.IntTensor([target.shape[0] for target in y_t])

        try:
            loss = ctc_loss(a_t, y_t, input_lengths, target_lengths)
            loss_item = loss.item()
            if math.isinf(loss_item):
                continue
            # print(loss_item)
            epoch_sub_losses.append(loss_item)
            loss.backward()
            optimizer.step()
        except:
            print("error")
            continue

        if epoch_i % 100 == 0:
            # https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/wave2letter.html
            d_a = a.clone().transpose(1, 2).detach().cpu().numpy()
            res = BeamSearch.ctcBeamSearch(d_a[0], classes, None, beamWidth=2048)
            print(res)

    # with open('./ckpt_acc_wav/losses/epoch_batch_losses_e_{}_b.pkl'.format(epoch_i),
    #           'wb') as handle:
    #     pickle.dump(epoch_sub_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # if epoch_i % 10 == 0:
    #     torch.save(wav2letter.state_dict(), "./ckpt_acc_wav/model/wav2letter_e_{}.ckpt".format(epoch_i))
