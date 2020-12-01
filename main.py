import pickle
from typing import List

import torch
import torchaudio
from torch import optim
from torch.utils.data import DataLoader
from transformers import ElectraTokenizer, ElectraModel

from criterion.Contrastive import ContrastiveLoss
from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate

train_on_gpu = torch.cuda.is_available()


def train_model_semantic(wav2vec: Wav2vecSemantic, optimizer: optim, epochs: int
                         , training_loader: DataLoader, test_loader: DataLoader, tokenizer
                         , semantic_model, batch_size) -> (Wav2vecSemantic, List):
    wav_model = wav2vec
    con_criterion = ContrastiveLoss()
    triplet_criterion = torch.nn.TripletMarginLoss()

    if train_on_gpu:
        wav_model.cuda()

    optimizer = optimizer

    epoch_mean_losses = []
    for epoch_i in range(epochs):

        # Enter training state
        wav_model.train()
        epoch_sub_losses = []
        batches_n = len(train_loader)
        for batch_i, (waveform, text_p) in enumerate(training_loader):
            # if batch for some reason fails
            print("b: {}/{}".format(batch_i, batches_n))

            if train_on_gpu:
                waveform = waveform.cuda()

            # Zero gradients
            optimizer.zero_grad()

            # Get electra embeddings as context
            # get random negative
            (_, text_n) = next(iter(training_loader))
            text_n = text_n[:len(text_p)]
            tokens = tokenizer([*text_p, *text_n], return_tensors="pt", padding=True)
            e = semantic_model(**tokens).last_hidden_state

            if train_on_gpu:
                e = e.cuda()

            embed_shape = e.shape[1]

            # Forward pass through architecture
            (hk, z, z_n), e_c = wav_model(x=waveform, idx_n=embed_shape)

            # Calculate contrastive loss / and triplet if text data
            loss_con = con_criterion(hk, z, z_n)
            loss_margin = triplet_criterion(e_c, e[:batch_size], e[batch_size:batch_size * 2])

            factor = 0.8
            loss = loss_margin + factor * loss_con
            # print(loss)

            epoch_sub_losses.append(loss.item())

            # Backprop
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            # if batch size is 256
            if batch_i % 20 == 0:
                # defrag GPU Mem
                with open('./ckpt/losses_batch/epoch_batch_losses_e_{}_b_{}.pkl'.format(epoch_i, batch_i),
                          'wb') as handle:
                    pickle.dump(epoch_sub_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
                torch.save(wav_model.state_dict(),
                           "./ckpt/model/semantic_256_e_{}_b_{}.ckpt".format(epoch_i, batch_i))

        print("e: {}/{}".format(epoch_i, epochs))
        torch.save(wav_model.state_dict(), "./ckpt/model/wav2vec_semantic_256_e_{}.ckpt".format(epoch_i))
        epoch_mean_losses.append(torch.tensor(epoch_sub_losses).mean().item())
        with open('./ckpt/losses_epoch/epoch_mean_losses_e_{}.pkl'.format(epoch_i), 'wb') as handle:
            pickle.dump(epoch_mean_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # TODO make some train and test metrics
        # wav_model.eval()

    return wav_model, epoch_mean_losses


if __name__ == "__main__":
    train_data = torchaudio.datasets.LIBRISPEECH("./data/", url="train-clean-100", download=True)
    test_data = torchaudio.datasets.LIBRISPEECH("./data/", url="test-clean", download=True)

    batch_size = 8
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True,
                              collate_fn=collate,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=True,
                             collate_fn=collate,
                             shuffle=False)

    # Define wav2vec model, optimizer and criterion
    wav_model = Wav2vecSemantic(channels=256)
    optimizer = torch.optim.Adam(wav_model.parameters(), lr=0.001)

    # Define electra model, loss and tokenizer
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator', return_dict=True)

    model, losses = train_model_semantic(wav2vec=wav_model,
                                         optimizer=optimizer,
                                         epochs=10,
                                         training_loader=train_loader,
                                         test_loader=test_loader,
                                         tokenizer=tokenizer,
                                         semantic_model=electra_model,
                                         batch_size=batch_size)
