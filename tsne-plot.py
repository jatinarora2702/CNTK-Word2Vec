from __future__ import absolute_import
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np
import pickle


dictpickle = 'w2v-dict.pkl'
embpickle = 'w2v-emb.pkl'


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


def main():
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    with open(embpickle, 'rb') as handle:
        final_embeddings = pickle.load(handle)

    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])

    with open(dictpickle, 'rb') as handle:
        dictionary = pickle.load(handle)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)


if __name__ == '__main__':
    main()
