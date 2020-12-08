import seaborn as sns
import torch
import torchaudio.models
from torch.utils.data import DataLoader

from models.Wav2LetterEmbed import Wav2LetterEmbed
from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate

sns.set()

train_on_gpu = torch.cuda.is_available()

if __name__ == "__main__":

    char2index = {
        " ": 0,
        "-": 1,
        **{chr(i + 96): i + 1 for i in range(1, 27)}
    }

    index2char = {
        0: " ",
        1: "-",
        **{i + 1: chr(i + 96) for i in range(1, 27)}
    }

    wav2letter = Wav2LetterEmbed(num_classes=len(char2index), num_features=256)

    wav_base = Wav2vecSemantic(channels=256, prediction_steps=6)
    wav_base.load_state_dict(
        torch.load("./ckpt_con_triplet/model/wav2vec_semantic_con_triplet_256_e_30.ckpt",
                   map_location=torch.device('cpu')))

    test_data = torchaudio.datasets.LIBRISPEECH("./data/", url="train-clean-100", download=True)
    batch_size = 2
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             pin_memory=True,
                             collate_fn=collate,
                             shuffle=False)

    if train_on_gpu:
        wav_base.cuda()

    if train_on_gpu:
        wav2letter.cuda()

    epochs = 10000

    ctc_loss = torch.nn.CTCLoss()
    optimizer = torch.optim.Adam(wav2letter.parameters(), lr=1e-4)

    wav_base.eval()
    wav2letter.train()

    for epoch_i in range(epochs):
        for i_batch, (wave, texts) in enumerate(test_loader):

            optimizer.zero_grad()

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

            y_t = torch.tensor(y)

            if train_on_gpu:
                y_t = y_t.cuda()

            if train_on_gpu:
                wave = wave.cuda()

            with torch.no_grad():
                _, c = wav_base(wave)

            a = wav2letter(c)
            a_t = a.transpose(1, 2).transpose(0, 1)

            input_lengths = torch.full((batch_size,), a_t.size()[0], dtype=torch.long)
            target_lengths = torch.IntTensor([target.shape[0] for target in y_t])

            loss = ctc_loss(a_t, y_t, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            # if i_batch % 100 == 0:
            print(loss)
            if i_batch % 400 == 0:
                max_probs = torch.argmax(a, 1)
                # print(loss)
                # print(y)
                print(max_probs)
                for max_probs_sentence in max_probs:
                    temp = ""
                    for index_char in max_probs_sentence:
                        if index_char is 0:
                            continue
                        temp += index2char[index_char.item()]
                    print(temp)
