from dataset import Recyclables
from sampler import PrototypicalBatchSampler
from protonet import ProtoNet
from proto_loss import prototypical_loss as loss_fn
from tqdm import tqdm
import torch
import numpy as np


def init_sampler(labels):
    return PrototypicalBatchSampler(labels=labels,
    classes_per_it=5, num_samples=5, iterations=100)


def init_dataloader():
    dataset = Recyclables()
    sampler = init_sampler(labels=dataset.labels)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return loader


def init_protonet():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = ProtoNet().to(device)
    return model


def init_optim(model):
    return torch.optim.Adam(params=model.parameters(), lr=0.001)


def train(tr_dataloader, model, optim):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    for epoch in range(100):
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=3)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        print(f'Loss: {np.mean(train_loss[-5:])}, Accuracy: {np.mean(train_acc[-5:])}')


model = init_protonet()
train(init_dataloader(), model, init_optim(model=model))