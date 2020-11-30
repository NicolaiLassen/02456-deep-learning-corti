import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import manifold

sns.set()


def TSNE_Wav2Vec_embed_Semantic_embed(X, batch_n=1):
    # TODO: REMOVE HARDCODE FOR TRANSFORMER TYPE
    pca = manifold.TSNE(perplexity=40, n_components=2, init='pca')
    X = pca.fit_transform(X)
    dtf = pd.DataFrame()

    dtf_group_1 = pd.DataFrame(X[0:batch_n], columns=["x", "y"])
    dtf_group_1["cluster"] = "wav2vec embed"

    dtf_group_2 = pd.DataFrame(X[batch_n:batch_n * 2], columns=["x", "y"])
    dtf_group_2["cluster"] = "electra embed pos"

    dtf_group_3 = pd.DataFrame(X[batch_n * 2:batch_n * 3], columns=["x", "y"])
    dtf_group_3["cluster"] = "electra embed neg"

    dtf = dtf.append(dtf_group_1)
    dtf = dtf.append(dtf_group_2)
    dtf = dtf.append(dtf_group_3)

    sns.scatterplot(data=dtf, x="x", y="y", hue="cluster")
    plt.show()
