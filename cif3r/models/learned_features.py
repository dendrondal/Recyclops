from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn.functional as F
import numpy as np


class TensorBoard:
    def __init__(self, loader, classes, net):
        self.writer = SummaryWriter("../../reports/UTK")
        self.images, self.labels = loader.next()
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
        grid = torchvision.utils.make_grid(self.images)
        _imshow(grid, one_channel=True)
        self.writer.add_image("image_sampling", grid)
        self.writer.close()

    def plot_embedding(self, n=100):
        perm = torch.randperm(len(self.images))
        imgs, lbls = self.images[perm][:n], self.labels[perm][:n]
        features = images.view(-1, 28*28)
        self.writer.add_embedding(features, 
                            metadata=lbls, 
                            label_img=imgs.unsqueeze(1))
        self.writer.close()

    def imgs_to_probs(self):
        output = self.net(self.images)
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

    def plot_class_preds(self):
        preds, probs = imgs_to_probs(self.net, self.images)
        fig = plt.figure(figsize=(12,48))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
            matplotlib_imshow(self.images[idx], one_channel=True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                self.classes[preds[idx]],
                probs[idx] * 100.0,
                self.classes[labels[idx]]),
                        color=("green" if preds[idx]==labels[idx].item() else "red"))
        self.writer.add_figure('predictions vs. actuals', fig)

    @staticmethod
    def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
        preds = test_preds == class_index
        probs = test_probs[:, class_index]
        writer.add_pr_curve(classes[class_index],
        preds, probs, global_step=global_step)
        writer.close()

    def plot_pr_curves(self, probs, preds):
        for i in range(len(self.classes)):
            self.add_pr_curve_tensorboard(i, probs, preds)
