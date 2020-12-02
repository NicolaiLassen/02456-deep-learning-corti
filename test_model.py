import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from transformers import ElectraTokenizer, ElectraModel

from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate

train_data = torchaudio.datasets.LIBRISPEECH("./data/", url="train-clean-100", download=True)

batch_size = 1
train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          collate_fn=collate,
                          shuffle=True)

wav_model = Wav2vecSemantic(channels=256)
wav_model.load_state_dict(torch.load("./ckpt/model/wav2vec_semantic_256_e_9.ckpt"))
wav_model.eval()

tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator', return_dict=True)

(wave, text_p) = next(iter(train_loader))
(_, text_n) = next(iter(train_loader))

print(text_p)
print(text_n)

tokens = tokenizer([*text_p, *text_n], return_tensors="pt", padding=True)
e = electra_model(**tokens).last_hidden_state

(_, _, _), ec = wav_model(wave, idx_n=len(e[0]))

print(ec.shape)
print(e[0].shape)

print(F.cosine_similarity(e[0], e[1]).sum())

print(F.cosine_similarity(ec[0], e[0]).sum())
print(F.cosine_similarity(ec[0], e[1]).sum())