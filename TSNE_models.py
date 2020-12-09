import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchaudio
from sklearn import manifold
from torch.utils.data import DataLoader

from models.Wav2VecSemantic import Wav2vecSemantic
from utils.training import collate

sns.set()

if __name__ == '__main__':
    model = Wav2vecSemantic(channels=256, prediction_steps=6)
    model.load_state_dict(
        torch.load("./ckpt_con_triplet/model/wav2vec_semantic_con_triplet_256_e_34.ckpt",
                   map_location=torch.device('cpu')))
    model.eval()

    test_data = torchaudio.datasets.LIBRISPEECH("./data/", url="test-clean", download=True)
    batch_size = 32
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             pin_memory=True,
                             collate_fn=collate,
                             shuffle=False)

    wave, text = next(iter(test_loader))

    z, c = model(wave)

    X = torch.stack([
        c.view(batch_size, -1),
        z.view(batch_size, -1),
    ]).view(batch_size * 2, -1).detach().cpu().numpy()

    pca = manifold.TSNE(n_components=2)
    X = pca.fit_transform(X)
    dtf = pd.DataFrame()

    dtf_group_1 = pd.DataFrame(X[0:batch_size], columns=["x", "y"])
    dtf_group_1["cluster"] = "wav2vec c"

    dtf_group_2 = pd.DataFrame(X[batch_size:batch_size * 2], columns=["x", "y"])
    dtf_group_2["cluster"] = "wav2vec z"

    dtf = dtf.append(dtf_group_1)
    dtf = dtf.append(dtf_group_2)

    sns.scatterplot(data=dtf, x="x", y="y", hue="cluster")
    plt.title("Contrastive Weights")
    plt.show()
