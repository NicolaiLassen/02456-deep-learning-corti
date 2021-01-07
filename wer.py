import argparse
import os
from typing import List

import seaborn as sns
import torch
import torchaudio
from torch.fft import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from models.Wav2LetterEmbed import Wav2LetterEmbed
from models.Wav2VecSemantic import Wav2vecSemantic
from results.scorer import get_asr_metric
from utils.decoder import CTCBeamDecoder
from utils.training import collate

sns.set()

train_on_gpu = torch.cuda.is_available()


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loss", default="con_triplet", help="verbose output")
    args = parser.parse_args()

    if args.loss not in ["con", "triplet", "con_triplet"]:
        exit(1)

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
    batch_size = 1
    epochs = 10000

    wav2letter = Wav2LetterEmbed(num_classes=len(labels), num_features=num_features)
    wav2letter.load_state_dict(
        torch.load("./ckpt_{}_wav2letter/model/{}_wav2letter.ckpt".format(args.loss, args.loss),
                   map_location=torch.device('cpu')
                   ),
    )
    wav_model = Wav2vecSemantic(channels=256, prediction_steps=6)
    wav_model.load_state_dict(
        torch.load("./ckpt_{}/model/wav2vec_semantic_{}_256_e_30.ckpt".format(args.loss, args.loss),
                   map_location=torch.device('cpu')))

    training_loader = DataLoader(dataset=train_data,
                                 batch_size=batch_size,
                                 pin_memory=True,
                                 collate_fn=collate,
                                 shuffle=False)
    decoder = CTCBeamDecoder("./lm/lm_librispeech_kenlm_word_4g_200kvocab.bin")


    def sentence_to_idx(sentence) -> List:
        y = []
        for char in sentence:
            try:
                y.append(char2index[char])
            except:
                continue
        return y


    def max_pad_batch_idx(transcripts) -> Tensor:
        with torch.no_grad():
            y = [torch.tensor(sentence_to_idx(sentence)) for sentence in transcripts]
            return pad_sequence(y, batch_first=True, padding_value=labels.index(blank))


    wav_model.eval()
    epoch_mean_losses = []
    for epoch_i in range(epochs):

        # Enter training state
        epoch_sub_losses = []
        wav2letter.eval()

        for wave, texts in training_loader:
            # try catch to fix last batch size
            y = max_pad_batch_idx(texts)

            with torch.no_grad():
                c, _ = wav_model(wave)

            target = "".join(texts)
            out = wav2letter(c)  # -> out (batch_size, number_of_classes, input_length).
            decoded = decoder(out.permute(0, 2, 1))  # <- beam in (batch_size, input_length, number_of_classes)
            print(decoded)
            print(get_asr_metric([target], [decoded]))

        break
