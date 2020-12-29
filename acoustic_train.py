import os
import pickle

import seaborn as sns
import torch
import torchaudio
# Linux gcc clang
from ctcdecode import CTCBeamDecoder
from torch.fft import Tensor
from torch.optim import Adam, lr_scheduler
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

    decoder = CTCBeamDecoder(
        labels,
        model_path="./lm/lm_librispeech_kenlm_word_4g_200kvocab.bin",
        alpha=0.522729216841,
        beta=0.96506699808,
        beam_width=1000,
        blank_id=labels.index(blank),
        log_probs_input=True
    )

    num_features = 256
    wav2letter = Wav2LetterEmbed(num_classes=len(char2index), num_features=num_features)

    wav_base = Wav2vecSemantic(channels=256, prediction_steps=6)
    wav_base.load_state_dict(
        torch.load("./ckpt_con_triplet/model/wav2vec_semantic_con_triplet_256_e_51.ckpt",
                   map_location=torch.device('cpu')))

    test_data = torchaudio.datasets.LIBRISPEECH("./data/", url="train-clean-100", download=True)
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

    criterion = torch.nn.CTCLoss(blank=labels.index(blank))
    optimizer = Adam(wav2letter.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.50, patience=6, verbose=True)

    wav_base.eval()
    wav2letter.train()


    def get_y_idxs(texts) -> Tensor:
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


    # TEMP
    wave, text = next(iter(test_loader))

    print("target: {}\n".format("".join(text)))

    for epoch_i in range(epochs):
        epoch_sub_losses = []
        # TODO
        # for batch_i, (wave, texts) in enumerate(test_loader):
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        y = get_y_idxs(text)

        if train_on_gpu:
            y = y.cuda()
            wave = wave.cuda()

        with torch.no_grad():
            _, c = wav_base(wave)

        out = wav2letter(c)  # -> out (batch_size, number_of_classes, input_length).

        out_p = out.permute(2, 0, 1)  # <- log_probs in (input_length, batch_size, number_of_classes)
        input_lengths = torch.full((batch_size,), fill_value=out_p.size(0), dtype=torch.int32)
        target_lengths = torch.full((batch_size,), fill_value=y.size(1), dtype=torch.int32)
        loss = criterion(out_p, y, input_lengths, target_lengths)

        loss_item = loss.item()
        print(loss_item)
        epoch_sub_losses.append(loss_item)

        loss.backward()
        optimizer.step()

        print(optimizer.param_groups[0]['lr'])
        scheduler.step(loss)

        with open('./ckpt_acc_wav/losses/epoch_batch_losses_e_{}_b.pkl'.format(epoch_i),
                  'wb') as handle:
            pickle.dump(epoch_sub_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if epoch_i % 10 == 0:
            with torch.no_grad():
                beam_results, beam_scores, timesteps, out_lens = decoder.decode(
                    out.permute(0, 2, 1))  # <- beam in (batch_size, input_length, number_of_classes)
                print("".join([index2char[n.item()] for n in beam_results[0][0][:out_lens[0][0]]]))

            torch.save(wav2letter.state_dict(), "./ckpt_acc_wav/model/wav2letter_e_{}.ckpt".format(epoch_i))
