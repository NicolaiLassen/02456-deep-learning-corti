import seaborn as sns

sns.set()
from sklearn import manifold


def TSNE_embed_context_plot(X):
    '''
    TSNE plot human observable context
    '''

    # TODO: What does the embeds show us - speech context?
    pca = manifold.TSNE(perplexity=40, n_components=2, init='pca')
    X = pca.fit_transform(X)
