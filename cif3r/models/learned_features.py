import pickle
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class TensorBoard:
    def __init__(self, loader, classes, net):
        self.writer = SummaryWriter("reports/UTK")
        self.loader = loader
        self.net = net
        self.classes = classes

    @staticmethod
    def _imshow(img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def write_grid(self):
        images, _ = iter(self.loader).next()
        grid = torchvision.utils.make_grid(images)
        self._imshow(grid)
        self.writer.add_image("image_sampling", grid)
        self.writer.close()

    def plot_embedding(self, n=100):
        images, labels = iter(self.loader).next()
        print(labels)
        perm = torch.randperm(len(images))
        imgs, lbls = images[perm][:n], labels[perm][:n]
        print(len(imgs), len(lbls))
        features = images.view(-1, 28 * 28)
        self.writer.add_embedding(features, metadata=lbls, label_img=imgs.unsqueeze(1))
        self.writer.close()

    def imgs_to_probs(self, images):
        output = self.net(images)
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

    def plot_class_preds(self):
        images, labels = iter(self.loader).next()
        preds, probs = self.imgs_to_probs(images)
        print(len(labels))
        fig = plt.figure(figsize=(12, 48))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
            self._imshow(images[idx])
            ax.set_title(
                "{0}, {1:.1f}%\n(label: {2})".format(
                    self.classes[labels[idx]],
                    probs[idx] * 100.0,
                    self.classes[labels[idx]],
                ),
                color=("green" if preds[idx] == labels[idx].item() else "red"),
            )
        self.writer.add_figure("predictions vs. actuals", fig)

    def add_pr_curve_tensorboard(
        self, class_index, test_probs, test_preds, global_step=0
    ):
        preds = test_preds == class_index
        probs = test_probs[:, class_index]
        self.writer.add_pr_curve(
            self.classes[class_index], preds, probs, global_step=global_step
        )
        self.writer.close()

    def plot_pr_curves(self, probs, preds):
        for i in range(len(self.classes)):
            self.add_pr_curve_tensorboard(i, probs, preds)


def load_embeddings():
    embeddings = []
    data_dir = Path(__file__).resolve().parents[2] / 'data/final'
    for file in data_dir.glob('*.pt'):
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
            y.append('trash')
    print(y)
    return X, y


def class_mappings():
    mapping = dict()
    dict_dir = Path(__file__).resolve().parents[2] / "data/external"
    with open(dict_dir / 'UTK.pickle', 'rb') as f:
        nested_dict = pickle.load(f)
    for k in nested_dict['R'].keys():
        for subcls in nested_dict['R'][k]:
            mapping[subcls] = k
    return mapping


def pca(X):
    clf  = PCA(n_components = 0.95)
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
    cmap = ['r', 'g', 'b', 'm']
    class_colors = {k: v for k, v in zip(list(set(y)), cmap)}
    for i, feature in enumerate(X):
        plt.scatter(
            features[i, 0],
            features[i, 1],
            marker = 'o',
            color=class_colors[y[i]],
            linewidth="1",
            alpha=0.8,
            label=y[i],
        )
    plt.legend(loc="best")
    plt.title("t-SNE on ProtoNet learned features")
    plt.savefig("/home/dal/CIf3R/reports/figures/protonet_tsne.png")


if __name__ == '__main__':
    X, y = embeddings_to_numpy(load_embeddings())
    lengths = [lst.shape for lst in X]
    X = pca(X)
    tsne(X, y)

