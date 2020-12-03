import pickle
import matplotlib.pyplot as plt
import torch
from models.Wav2VecSemantic import Wav2vecSemantic
from models.Wav2Vec import Wav2vec
import torchaudio
from torch.utils.data import DataLoader
from utils.training import collate
from criterion.CosineDist import CosineDistLoss


if __name__ == "__main__":

    # Create models and load their weights - basic wav2vec and with text data
    wav_base = Wav2vecSemantic(channels=256)
    wav_base.load_state_dict(torch.load("./ckpt_base_wav2vec/model/wav2vec_semantic_256_e_9.ckpt", map_location=torch.device('cpu')))
    wav_base.eval()

    wav_semantic = Wav2vecSemantic(channels=256)
    wav_semantic.load_state_dict(torch.load("./ckpt/model/wav2vec_semantic_256_e_9.ckpt", map_location=torch.device('cpu')))
    wav_semantic.eval()

    # Dataset for testing
    test_data = torchaudio.datasets.LIBRISPEECH("./data/", url="test-clean", download=True)
    batch_size = 64
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             pin_memory=True,
                             collate_fn=collate,
                             shuffle=False)

    # Loss comparison
    loss = CosineDistLoss()

    (waveform, text_p) = next(iter(test_loader))
    _, _, _, z_base, c_base = wav_base(x=waveform)
    _, z_sem, c_sem = wav_semantic(x=waveform, idx_n=None, use_semantic=False)
    
    print(c_base)
    print(c_sem)



"""
    for batch_i, (waveform, text_p) in enumerate(test_loader):
        _, _, _, z_base, c_base = wav_base(x=waveform)
        _, z_sem, c_sem = wav_semantic(x=waveform, idx_n=None, use_semantic=False)
        
"""






