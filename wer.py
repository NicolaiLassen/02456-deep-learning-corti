import argparse
import os

import seaborn as sns
import torch
import torchaudio
from torch.fft import Tensor
from torch.utils.data import DataLoader

from models.Wav2LetterEmbed import Wav2LetterEmbed
from models.Wav2VecSemantic import Wav2vecSemantic
from ctcdecode import CTCBeamDecoder
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
        torch.load("./ckpt_{}_wav2letter/model/{}_wav2letterss.ckpt".format(args.loss, args.loss),
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

    decoder = CTCBeamDecoder(
        labels,
        model_path="./lm/lm_librispeech_kenlm_word_4g_200kvocab.bin",
        alpha=0.522729216841,
        beta=0.66506699808,
        beam_width=60000,
        blank_id=labels.index(blank),
        log_probs_input=True
    )


    def max_pad_batch_idx(transcripts) -> Tensor:
        with torch.no_grad():
            y = []
            len_max = len(max(transcripts, key=len))
            for i, sentence in enumerate(transcripts):
                y.append([])
                for char in sentence.lower():
                    try:
                        y[i].append(char2index[char])
                    except Exception as e:
                        print(e)
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
        wav2letter.eval()

        for wave, texts in training_loader:
            # try catch to fix last batch size
            y = max_pad_batch_idx(texts)

            with torch.no_grad():
                c, _ = wav_model(wave)

            out = wav2letter(c)  # -> out (batch_size, number_of_classes, input_length).

            beam_results, beam_scores, timesteps, out_lens = decoder.decode(
                out.permute(0, 2, 1))  # <- beam in (batch_size, input_length, number_of_classes)
            # First sentence
            print("target:", "".join(texts[0]))
            print("Log probs:", "".join([index2char[n.item()] for n in beam_results[0][0][:out_lens[0][0]]]))

        break