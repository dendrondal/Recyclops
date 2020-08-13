import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_embeddings():
    embeddings = []
    data_dir = Path(__file__).resolve().parents[1] / "data/final"
    for file in data_dir.glob("*.pt"):
        embeddings.append(torch.load(file))
    return embeddings


def embeddings_to_numpy(embedding_list):
    X = [list(d.values())[0].detach().numpy().reshape(1600,) for d in embedding_list]
    subclasses = [str(list(d.keys())[0]) for d in embedding_list]
    maps = class_mappings()
    y = []
    for i in subclasses:
        try:
            y.append(maps[i])
        except KeyError:
            y.append("trash")
    return X, y


def class_mappings():
    mapping = dict()
    dict_dir = Path(__file__).resolve().parents[1] / "data/external"
    with open(dict_dir / "UTK.pickle", "rb") as f:
        nested_dict = pickle.load(f)
    for k in nested_dict["R"].keys():
        for subcls in nested_dict["R"][k]:
            mapping[subcls] = k
    return mapping


def pca(X):
    clf = PCA(n_components=0.95)
    clf.fit(X)
    return clf.transform(X)


def tsne(X, y):
    tsne_obj = TSNE(
        n_components=2,
        init="pca",
        random_state=101,
        method="barnes_hut",
        n_iter=500,
        verbose=2,
        n_jobs=-1,
    )
    features = tsne_obj.fit_transform(X)

    plt.figure(figsize=(10, 10))
    cmap = ["r", "g", "b", "m"]
    class_colors = {k: v for k, v in zip(list(set(y)), cmap)}
    for i, feature in enumerate(X):
        plt.scatter(
            features[i, 0],
            features[i, 1],
            marker="o",
            color=class_colors[y[i]],
            linewidth="1",
            alpha=0.8,
            label=y[i],
        )
    plt.legend(loc="best")
    plt.title("t-SNE on ProtoNet learned features")
    plt.savefig("/home/dal/CIf3R/reports/figures/protonet_tsne.png")


if __name__ == "__main__":
    X, y = embeddings_to_numpy(load_embeddings())
    lengths = [lst.shape for lst in X]
    X = pca(X)
    tsne(X, y)
