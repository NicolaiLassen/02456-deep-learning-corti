import argparse
import os
import pickle
from typing import List

import torch
import torchaudio
from torch import optim
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from transformers import ElectraTokenizer, ElectraModel

from criterion.ContrastiveWav import ContrastiveLoss
from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate

train_on_gpu = torch.cuda.is_available()


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def train_model_semantic(wav_model: Wav2vecSemantic,
                         optimizer: optim,
                         scheduler: lr_scheduler,
                         epochs: int,
                         args: argparse.Namespace,
                         training_loader: DataLoader,
                         tokenizer,
                         semantic_model) -> (Wav2vecSemantic, List):
    create_dir("./ckpt_{}".format(args.loss))
    create_dir("./ckpt_{}/losses_batch".format(args.loss))
    create_dir("./ckpt_{}/losses_epoch".format(args.loss))
    create_dir("./ckpt_{}/model".format(args.loss))

    # context weight
    alpha = 0.4

    # define loss
    con_criterion = ContrastiveLoss()
    triplet_criterion = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=torch.nn.PairwiseDistance())

    if train_on_gpu:
        wav_model.cuda()

    epoch_mean_losses = []
    print("loss:", args.loss)
    for epoch_i in range(epochs):

        # Enter training state
        wav_model.train()
        epoch_sub_losses = []

        for wave, texts_p in training_loader:

            # defrag GPU Mem
            torch.cuda.empty_cache()

            if train_on_gpu:
                wave = wave.cuda()

            # Zero gradients
            optimizer.zero_grad()

            # Select training type
            loss = None
            if args.loss == "con":
                hk, z, z_n = wav_model(x=wave, contrastive_train=True)
                loss = con_criterion(hk, z, z_n)

            if args.loss == "triplet" or \
                    args.loss == "con_triplet":

                batch_length = len(texts_p)

                (_, texts_n) = next(iter(training_loader))

                texts_n = texts_n[:batch_length]
                texts_p = texts_p[:batch_length]

                tokens = tokenizer([*texts_p, *texts_n], return_tensors="pt", padding=True)
                e_embed = semantic_model(**tokens).last_hidden_state

                if train_on_gpu:
                    e_embed = e_embed.cuda()

                e_embed_p = e_embed[:batch_length].permute(1, 0, 2)
                e_embed_n = e_embed[batch_length:batch_length * 2].permute(1, 0, 2)

                if args.loss is "triplet":
                    z_embed = wav_model(x=wave, contrastive_train=False, context=e_embed_p)
                    loss = triplet_criterion(z_embed, e_embed_p, e_embed_n)
                else:
                    (hk, z, z_n), z_embed = wav_model(x=wave, contrastive_train=True, context=e_embed_p)
                    loss_con = con_criterion(hk, z, z_n)
                    loss_triplet = triplet_criterion(z_embed, e_embed_p, e_embed_n)
                    loss = loss_con + loss_triplet * alpha

            # Backprop
            loss.backward()
            print(loss)
            optimizer.step()
            # lower the lr if the alg is stuck
            scheduler.step(loss)
            # graph
            epoch_sub_losses.append(loss.item())

        epoch_mean_losses.append(torch.tensor(epoch_sub_losses).mean().item())

        with open('./ckpt_{}/losses_batch/epoch_batch_losses_e_{}_b.pkl'.format(args.loss, epoch_i),
                  'wb') as handle:
            pickle.dump(epoch_sub_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('./ckpt_{}/losses_epoch/epoch_mean_losses_e_{}.pkl'.format(args.loss, epoch_i), 'wb') as handle:
            pickle.dump(epoch_mean_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

        torch.save(wav_model.state_dict(), "./ckpt_{}/model/wav2vec_semantic_{}_256_e_{}.ckpt".format(args.loss,
                                                                                                      args.loss,
                                                                                                      epoch_i))


if __name__ == "__main__":
    train_data = torchaudio.datasets.LIBRISPEECH("./data/", url="train-clean-100", download=True)
    test_data = torchaudio.datasets.LIBRISPEECH("./data/", url="test-clean", download=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loss", default="con", help="Pick the loss and the traning method")
    args = parser.parse_args()

    if args.loss not in ["con", "triplet", "con_triplet"]:
        exit(1)

    batch_size = 2
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              pin_memory=True,
                              collate_fn=collate,
                              shuffle=True)

    # Define wav2vec model, optimizer, lr_scheduler and criterion
    wav_model = Wav2vecSemantic(channels=256, prediction_steps=6)
    optimizer = Adam(wav_model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.50, patience=10)

    # Define electra model and tokenizer
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator', return_dict=True)

    train_model_semantic(wav_model=wav_model,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         epochs=100,
                         args=args,
                         training_loader=train_loader,
                         tokenizer=tokenizer,
                         semantic_model=electra_model)
