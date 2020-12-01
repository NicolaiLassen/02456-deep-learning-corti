import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import ElectraTokenizer

from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate

train_data = torchaudio.datasets.LIBRISPEECH("./data/", url="train-clean-100", download=True)

batch_size = 4
train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          collate_fn=collate,
                          shuffle=True)

wav_model = Wav2vecSemantic(channels=256)
wav_model.load_state_dict(torch.load("./ckpt/model/semantic_256_e_0_b_7000.ckpt"))
wav_model.eval()

(wave, text_n) = next(iter(train_loader))

wav_model(wave)
