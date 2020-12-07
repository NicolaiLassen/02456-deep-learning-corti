import torch
import torchaudio.models
from torch.utils.data import DataLoader
from criterion.Contrastive import ContrastiveLoss
from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate, GreedyDecoder


if __name__ == "__main__":

    test_data = torchaudio.datasets.LIBRISPEECH("./data/", url="test-clean", download=True)

    model_base = Wav2vecSemantic(channels=256, prediction_steps=6)
    model_trip = Wav2vecSemantic(channels=256, prediction_steps=6)
    model_con_trip = Wav2vecSemantic(channels=256, prediction_steps=6)

    model_base.load_state_dict(torch.load("./ckpt_con/mode/wav2vec_semantic_con_256_e_34.ckpt", map_location=torch.device('cpu')))
    model_trip.load_state_dict(torch.load("./ckpt_triplet/model/wav2vec_semantic_triplet_256_e_52.ckpt", map_location=torch.device('cpu')))
    model_con_trip.load_state_dict(torch.load("./ckpt_con_triplet/model/wav2vec_semantic_con_triplet_256_e_51.ckpt", map_location=torch.device('cpu')))

    model_base.eval()
    model_trip.eval()
    model_con_trip.eval()

    batch_size = 64
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             pin_memory=True,
                             collate_fn=collate,
                             shuffle=False)
    train_on_gpu = False
    if train_on_gpu:
        model_base.cuda()
        model_trip.cuda()
        model_con_trip.cuda()


    loss_base = []
    loss_trip = []
    loss_con_trip = []

    con_criterion = ContrastiveLoss()
    for waveform, text_p in test_loader:

        # defrag GPU Mem
        torch.cuda.empty_cache()

        if train_on_gpu:
            waveform = waveform.cuda()

        # Base loss
        hk, z, z_n = model_base(x=waveform)
        loss_base.append(con_criterion(hk, z, z_n))

        # Triplet
        hk, z, z_n = model_trip(x=waveform)
        loss_trip.append(con_criterion(hk, z, z_n))

        # Con + trip
        hk, z, z_n = model_trip(x=waveform)
        loss_con_trip.append(con_criterion(hk, z, z_n))
