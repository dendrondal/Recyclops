from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn.functional as F
import numpy as np


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
        features = images.view(-1, 28*28)
        self.writer.add_embedding(features, 
                            metadata=lbls, 
                            label_img=imgs.unsqueeze(1))
        self.writer.close()

    def imgs_to_probs(self, images):
        output = self.net(images)
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

    def plot_class_preds(self):
        images, labels = iter(self.loader).next()
        preds, probs = self. imgs_to_probs(images)
        print(len(labels))
        fig = plt.figure(figsize=(12,48))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
            self._imshow(images[idx])
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                self.classes[labels[idx]],
                probs[idx] * 100.0,
                self.classes[labels[idx]]),
                        color=("green" if preds[idx]==labels[idx].item() else "red"))
        self.writer.add_figure('predictions vs. actuals', fig)


    def add_pr_curve_tensorboard(self, class_index, test_probs, test_preds, global_step=0):
        preds = test_preds == class_index
        probs = test_probs[:, class_index]
        self.writer.add_pr_curve(self.classes[class_index],
        preds, probs, global_step=global_step)
        self.writer.close()

    def plot_pr_curves(self, probs, preds):
        for i in range(len(self.classes)):
            self.add_pr_curve_tensorboard(i, probs, preds)