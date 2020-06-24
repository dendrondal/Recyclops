from dataset import Recyclables
from sampler import PrototypicalBatchSampler
from protonet import ProtoNet
from proto_loss import prototypical_loss as loss_fn
from learned_features import te
from parser_util import get_parser
from tqdm import tqdm
from pathlib import Path
import torch
import numpy as np


def init_seed(opt):
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt):
    dataset = Recyclables(opt.university, opt.num_support_tr + opt.num_query_tr)
    n_classes = len(np.unique(dataset.labels))
    print(n_classes)
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise (
            Exception(
                "There are not enough classes in the dataset in order "
                + "to satisfy the chosen classes_per_it. Decrease the "
                + "classes_per_it_{tr/val} option and try again."
            )
        )
    return dataset


def init_sampler(opt, labels, mode):
    if "train" in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(
        labels=labels,
        classes_per_it=classes_per_it,
        num_samples=num_samples,
        iterations=opt.iterations,
    )


def init_dataloader(opt, mode):
    dataset = init_dataset(opt)
    sampler = init_sampler(opt, dataset.labels, mode)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return loader


def init_protonet():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ProtoNet().to(device)
    return model


def init_optim(opt, model):
    return torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    """
    Initialize the learning rate scheduler
    """
    return torch.optim.lr_scheduler.StepLR(
        optimizer=optim, gamma=opt.lr_scheduler_gamma, step_size=opt.lr_scheduler_step
    )


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None, board):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    model_dir = Path(__file__).resolve().parents[2] / "models"
    last_model_path = model_dir / f"{opt.university}_last_model.pth"
    best_model_path = model_dir / f"{opt.university}_best_model.pth"

    for epoch in range(opt.epochs):
        print(f"===== Epoch {epoch} =====")
        tr_iter = iter(tr_dataloader)
        model.train()

        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y, n_support=opt.num_support_val)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())

        avg_loss = np.mean(train_loss[-opt.iterations :])
        avg_acc = np.mean(train_acc[-opt.iterations :])
        print(f"Loss: {avg_loss}, Accuracy: {avg_acc}")
        board.writer.add_scalar('training loss', avg_loss, epoch)
        board.writer.add_scalar('training accuracy', avg_acc, epoch)
        lr_scheduler.step()

        if val_dataloader is None:
            continue

        val_iter = iter(val_dataloader)
        model.eval()
        class_probs = []
        class_preds = []
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y, n_support=opt.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
            #Callbacks for PR curve
            class_probs_batch = [F.softmax(el, dim=0) for el in model_output]
            _, class_preds_batch = torch.max(output, 1)
            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)
        
        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)
        board.plot_pr_curves(test_probs, test_preds)
        avg_loss = np.mean(val_loss[-opt.iterations :])
        avg_acc = np.mean(val_acc[-opt.iterations :])
        board.writer.add_scalar('val loss', avg_loss, epoch)
        board.writer.add_scalar('val accuracy', avg_acc, epoch)
        postfix = " (Best)" if avg_acc >= best_acc else " (Best: {})".format(best_acc)
        print("Avg Val Loss: {}, Avg Val Acc: {}{}".format(avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ["train_loss", "train_acc", "val_loss", "val_acc"]:
        save_list_to_file(model_dir / f"{name}.txt", locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):
    """
    Test the model trained with the prototypical learning algorithm
    """
    device = "cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu"
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y, n_support=opt.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print("Test Acc: {}".format(avg_acc))

    return avg_acc


def main():
    """
    Initialize everything and train
    """
    options = get_parser().parse_args()

    init_seed(options)

    tr_dataloader = init_dataloader(options, "train")
    val_dataloader = init_dataloader(options, "val")
    test_dataloader = init_dataloader(options, "test")

    model = init_protonet()
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(
        opt=options,
        tr_dataloader=tr_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optim=optim,
        lr_scheduler=lr_scheduler,
    )
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print("Testing with last model..")
    test(opt=options, test_dataloader=test_dataloader, model=model)

    model.load_state_dict(best_state)
    print("Testing with best model..")
    test(opt=options, test_dataloader=test_dataloader, model=model)


if __name__ == "__main__":
    main()
