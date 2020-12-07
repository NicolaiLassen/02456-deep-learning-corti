import seaborn as sns
import torch
import torchaudio.models
from torch.utils.data import DataLoader

from models.Wav2Letter import Wav2LetterWithFit
from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate, GreedyDecoder

sns.set()

if __name__ == "__main__":
    # Create models and load their weights - basic wav2vec and with text data

    wav2letter = Wav2LetterWithFit(num_classes=40, input_type='mfcc', num_features=256)

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

    ctc_loss = torch.nn.CTCLoss()
    optimizer = torch.optim.Adam(wav2letter.parameters(), lr=1e-4)

    (wave, text) = next(iter(test_loader))
    z, c = wav_base(wave)
    print(c.size())

    out = wav2letter.fit(wave, torch.tensor(text), optimizer, ctc_loss, batch_size, epoch=1)
    print(out.shape)

    output = GreedyDecoder(out)

    print("sample target", text)
    print("predicted", output)

    #
    # c = c.view(batch_size,-1)
    # z = z.view(batch_size,-1)
    #
    # X= torch.stack([c, z]).view(batch_size * 2, -1).detach().numpy()
    #
    # pca = manifold.TSNE(perplexity=40, n_components=2, n_iter_without_progress=20000, n_iter=20000)
    # X = pca.fit_transform(X)
    # dtf = pd.DataFrame()
    #
    # dtf_group_1 = pd.DataFrame(X[0:batch_size], columns=["x", "y"])
    # dtf_group_1["cluster"] = "wav2vec c"
    #
    # dtf_group_2 = pd.DataFrame(X[batch_size:batch_size * 2], columns=["x", "y"])
    # dtf_group_2["cluster"] = "wav2vec z"
    #
    # dtf = dtf.append(dtf_group_1)
    # dtf = dtf.append(dtf_group_2)
    #
    # sns.scatterplot(data=dtf, x="x", y="y", hue="cluster")
    # plt.title("Weights Triplet")
    # plt.show()

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
