import argparse
import os
import pickle

import seaborn as sns
import torch
import torchaudio
from torch.fft import Tensor
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

from models.Wav2LetterEmbed import Wav2LetterEmbed
from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate

sns.set()

train_on_gpu = torch.cuda.is_available()


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loss", default="con", help="verbose output")
    args = parser.parse_args()

    if args.loss not in ["con", "triplet", "con_triplet"]:
        exit(1)

    create_dir("./ckpt_{}_wav2letter".format(args.loss))
    create_dir("./ckpt_{}_wav2letter/losses_epoch".format(args.loss))
    create_dir("./ckpt_{}_wav2letter/model".format(args.loss))

    train_data = torchaudio.datasets.LIBRISPEECH("./data/", url="train-clean-100", download=True)

    blank = "-"

    labels = [
        " ",
        *[chr(i + 96) for i in range(1, 27)],
        blank
    ]

    char2index = {
        **{label: i for i, label in enumerate(labels)},
    }

    index2char = {
        **{i: label for i, label in enumerate(labels)},
    }

    lr = 1e-4
    num_features = 256
    batch_size = 8
    epochs = 10000

    wav2letter = Wav2LetterEmbed(num_classes=len(labels), num_features=num_features)
    wav_model = Wav2vecSemantic(channels=256, prediction_steps=6)
    wav_model.load_state_dict(
        torch.load("./ckpt_{}/model/wav2vec_semantic_{}_256_e_30.ckpt".format(args.loss, args.loss),
                   map_location=torch.device('cpu')))

    training_loader = DataLoader(dataset=train_data,
                                 batch_size=batch_size,
                                 pin_memory=True,
                                 collate_fn=collate,
                                 shuffle=False)

    if train_on_gpu:
        wav_model.cuda()
        wav2letter.cuda()

    criterion = torch.nn.CTCLoss(blank=labels.index(blank), zero_infinity=True)
    optimizer = AdamW(wav2letter.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.50, patience=6)


    def max_pad_batch_idx(transcripts) -> Tensor:
        with torch.no_grad():
            y = []
            len_max = len(max(transcripts, key=len))
            for i, sentence in enumerate(transcripts):
                y.append([])
                for char in sentence.lower():
                    try:
                        y[i].append(char2index[char])
                    except:
                        continue
                # pad to longest
                y[i] += [labels.index(blank)] * (len_max - len(y[i]))

            # return int16 tensor
            return torch.tensor(y, dtype=torch.int16)


    wav_model.eval()
    epoch_mean_losses = []
    for epoch_i in range(epochs):

        # Enter training state
        epoch_sub_losses = []
        wav2letter.train()

        for wave, texts in training_loader:

            optimizer.zero_grad()

            torch.cuda.empty_cache()

            y = max_pad_batch_idx(texts)

            if train_on_gpu:
                y, wave = y.cuda(), wave.cuda()

            with torch.no_grad():
                c, _ = wav_model(wave)

            out = wav2letter(c)  # -> out (batch_size, number_of_classes, input_length).

            out_p = out.permute(2, 0, 1)  # <- log_probs in (input_length, batch_size, number_of_classes)
            input_lengths = torch.full((batch_size,), fill_value=out_p.size(0), dtype=torch.int32)
            target_lengths = torch.full((batch_size,), fill_value=y.size(1), dtype=torch.int32)
            # CTC loss
            loss = criterion(out_p, y, input_lengths, target_lengths)

            # Backprop
            loss.backward()
            # print(loss) # test if it works
            optimizer.step()
            # lower the lr if the alg is stuck
            scheduler.step(loss)
            # print(loss_item)
            # graph
            epoch_sub_losses.append(loss.item())

        epoch_mean_losses.append(torch.tensor(epoch_sub_losses).mean().item())

        with open('./ckpt_{}_wav2letter/losses_epoch/epoch_mean_losses_e_{}.pkl'.format(args.loss, epoch_i),
                  'wb') as handle:
            pickle.dump(epoch_mean_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

        torch.save(wav2letter.state_dict(),
                   "./ckpt_{}_wav2letter/model/{}_wav2letter_e_{}.ckpt".format(args.loss, args.loss, epoch_i))
