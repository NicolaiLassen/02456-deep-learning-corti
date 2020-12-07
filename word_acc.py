import seaborn as sns
import torch
import torchaudio.models
from torch.utils.data import DataLoader

from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate

sns.set()

if __name__ == "__main__":

    grapheme_count = 26 + 1
    wav2letter = torchaudio.models.Wav2Letter(num_classes=grapheme_count, input_type='mfcc', num_features=256)

    wav_base = Wav2vecSemantic(channels=256, prediction_steps=6)
    wav_base.load_state_dict(
        torch.load("./ckpt_con/model/wav2vec_semantic_con_256_e_30.ckpt", map_location=torch.device('cpu')))

    wav_base.eval()
    test_data = torchaudio.datasets.LIBRISPEECH("./data/", url="train-clean-100", download=True)
    batch_size = 1
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             pin_memory=True,
                             collate_fn=collate,
                             shuffle=False)


    epochs = 10

    ctc_loss = torch.nn.CTCLoss()
    optimizer = torch.optim.Adam(wav2letter.parameters(), lr=1e-4)
    for epoch_i in range(epochs):
        for wave, text in test_loader:
            z, c = wav_base(wave)
            a = wav2letter(c)
            print(a)
