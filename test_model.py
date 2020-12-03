import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="con", help="verbose output")
    args = parser.parse_args()


    print(args.model)

    # # Create models and load their weights - basic wav2vec and with text data
    # wav_base = Wav2vecSemantic(channels=256)
    # wav_base.load_state_dict(torch.load("./ckpt_base_wav2vec/model/wav2vec_semantic_256_e_9.ckpt", map_location=torch.device('cpu')))
    # wav_base.eval()
    #
    # wav_semantic = Wav2vecSemantic(channels=256)
    # wav_semantic.load_state_dict(torch.load("./ckpt/model/wav2vec_semantic_256_e_9.ckpt", map_location=torch.device('cpu')))
    # wav_semantic.eval()
    #
    # # Dataset for testing
    # """
    # test_data = torchaudio.datasets.LIBRISPEECH("./data/", url="test-clean", download=True)
    # batch_size = 1
    # test_loader = DataLoader(dataset=test_data,
    #                          batch_size=batch_size,
    #                          pin_memory=True,
    #                          collate_fn=collate,
    #                          shuffle=False)
    # """
    #
    # train_data = torchaudio.datasets.LIBRISPEECH("./data/", url="train-clean-100", download=True)
    # batch_size = 1
    # train_loader = DataLoader(dataset=train_data,
    #                               batch_size=batch_size,
    #                               pin_memory=True,
    #                               collate_fn=collate,
    #                               shuffle=True)
    # # Loss comparison
    # loss = CosineDistLoss()
    #
    # for name, param in wav_base.named_parameters():
    #     if name == "prediction.transpose_context.weight":
    #         print(name, param.data)
