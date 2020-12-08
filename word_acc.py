import seaborn as sns
import torch
import torchaudio.models
from torch.utils.data import DataLoader

from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate

sns.set()

train_on_gpu = torch.cuda.is_available()

if __name__ == "__main__":

    char2index = {
        " ": 0,
        "pad": 1,
        **{chr(i + 96): i + 1 for i in range(1, 27)}
    }
    index2char = {
        0: " ",
        1: "pad",
        **{i + 1: chr(i + 96) for i in range(1, 27)}
    }

    grapheme_count = 26 + 1
    wav2letter = torchaudio.models.Wav2Letter(num_classes=grapheme_count, input_type='mfcc', num_features=256)

    wav_base = Wav2vecSemantic(channels=256, prediction_steps=6)
    wav_base.load_state_dict(
        torch.load("./ckpt_con/model/wav2vec_semantic_con_256_e_30.ckpt", map_location=torch.device('cpu')))

    wav_base.eval()
    test_data = torchaudio.datasets.LIBRISPEECH("./data/", url="train-clean-100", download=True)
    batch_size = 4
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             pin_memory=True,
                             collate_fn=collate,
                             shuffle=False)

    if train_on_gpu:
        wav_base.cuda()

    if train_on_gpu:
        wav2letter.cuda()

    epochs = 10

    ctc_loss = torch.nn.CTCLoss()
    optimizer = torch.optim.Adam(wav2letter.parameters(), lr=1e-4)

    wav2letter.train()
    for epoch_i in range(epochs):
        for i_batch, (wave, text) in enumerate(test_loader):

            optimizer.zero_grad()

            y = []
            len_max = len(max(text, key=len))
            for i, sentence in enumerate(text):
                y.append([])
                for char in sentence.lower():
                    try:
                        y[i].append(char2index[char])
                    except:
                        continue
                y[i] += [1] * (len_max - len(y[i]))

            y = torch.tensor(y)
            if train_on_gpu:
                y = y.cuda()

            if train_on_gpu:
                wave = wave.cuda()

            _, c = wav_base(wave)
            a = wav2letter(c)

            # get ready to ctc
            a_t = a.transpose(1, 2).transpose(0, 1)
            input_lengths = torch.full(size=(a_t.size()[1],), fill_value=a_t.size()[0], dtype=torch.long)
            target_lengths = torch.IntTensor([target.shape[0] for target in y])

            loss = ctc_loss(a_t, y, input_lengths, target_lengths)

            loss.backward()

            if i_batch % 100 == 0:
                print(i_batch)
                print(loss)
                a_g = a.argmax(dim=1)
                for sentence in a_g:
                    char_sentence = ""
                    for index in sentence:
                        char_sentence += index2char[index.item()]
                    print(char_sentence)
