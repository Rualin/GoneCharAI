import os
import multiprocessing

import numpy as np
import torch
import torchvision
import data_preprocess as dp
import matplotlib.pyplot as plt
import torchmetrics.classification as tmc
from tqdm import tqdm
from torch import nn

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", True)
torch.set_default_device(dp.DEVICE)
BS = 32
EPOCHES = 30
LR = 0.001
# LRS = [0.01, 0.005, 0.001, 0.0005]
LRS = [0.1, 0.01, 0.005]
PATH = 'saves/save_no_onehot.pth'
NUM_WORKERS = 20


def train_val_loop(model, criterion, optimizer, dl, is_val):
    if is_val:
        model.eval()
    else:
        model.train()
    running_loss = 0.0
    mf1s = tmc.MulticlassF1Score(17)
    macc = tmc.MulticlassAccuracy(17)
    mprec = tmc.MulticlassPrecision(17)
    mrec = tmc.MulticlassRecall(17)
    for i, data in enumerate(tqdm(dl, desc="Iterations")):
        inputs = data[0].to(dp.DEVICE)
        labels = data[1].to(dp.DEVICE)
        if not(is_val):
            optimizer.zero_grad()
        outputs:torch.Tensor = model(inputs)
        loss = criterion(outputs, labels)
        if not(is_val):
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        mf1s.update(outputs, labels)
        macc.update(outputs, labels)
        mprec.update(outputs, labels)
        mrec.update(outputs, labels)
    if is_val:
        print("\tValid:")
    else:
        print("\tTrain:")
    print(f'\t\tLoss: {(running_loss / len(dl)):.3f}, acc: {macc.compute():.3f}, f1: {mf1s.compute():.3f}, prec: {mprec.compute():.3f}, rec: {mrec.compute():.3f}')
    return running_loss / len(dl), macc.compute().cpu(), mf1s.compute().cpu()


def training(model, criterion, lrs, train_dl, val_dl, epoches=10):
    '''
    Returns train_losses, val_losses, train_f1, val_f1, train_acc, val_acc
    '''
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    train_accs = []
    val_accs = []
    maxf1 = 0
    lr_id = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lrs[lr_id])
    for epoch in range(epoches):
        if (epoch % 10 == 0) and (epoch != 0):
            lr_id = min(len(lrs) - 1, lr_id + 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=lrs[lr_id])
        print("Epoch:", epoch)
        tr_loss, tr_acc, tr_f1 = train_val_loop(model, criterion, optimizer, train_dl, False)
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        train_f1s.append(tr_f1)

        val_loss, val_acc, val_f1 = train_val_loop(model, criterion, optimizer, val_dl, True)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        if val_f1 > maxf1:
            if maxf1 != 0:
                os.system(f"rm saves/maxf1_{(maxf1 * 100):.0f}.pth")
            maxf1 = val_f1
            torch.save(model.state_dict(), f"saves/maxf1_{(val_f1 * 100):.0f}.pth")
        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), f"saves/epoch_{epoch}.pth")
            plot = metrics_plot(train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs)
            if not(os.path.exists("plots")):
                os.mkdir("plots")
            plot.savefig(f"plots/MetricsResNet50_{epoch}Epoches.png")
    return train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs

def metrics_plot(train_losses, val_losses, train_f1, val_f1, train_acc, val_acc):
    fig, ax = plt.subplots(nrows=3, ncols=2)
    # ax.figure("Metrics", figsize=(10, 10))
    # ax.subplot(3, 2, 1)
    ax[0, 0].plot(train_losses)
    ax[0, 0].set_title("Train loss")
    # plt.subplot(3, 2, 2)
    ax[0, 1].plot(val_losses)
    ax[0, 1].set_title("Valid loss")
    # plt.subplot(3, 2, 3)
    ax[1, 0].plot(train_f1)
    ax[1, 0].set_title("Train f1")
    # plt.subplot(3, 2, 4)
    ax[1, 1].plot(val_f1)
    ax[1, 1].set_title("Valid f1")
    # plt.subplot(3, 2, 5)
    ax[2, 0].plot(train_acc)
    ax[2, 0].set_title("Train acc")
    # plt.subplot(3, 2, 6)
    ax[2, 1].plot(val_acc)
    ax[2, 1].set_title("Valid acc")
    fig.suptitle("Metrics")
    return fig


if __name__ == "__main__":
    print(dp.DEVICE)
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to(device=dp.DEVICE)
    print(model.fc)
    model.fc = torch.nn.Linear(2048, 17, bias=True)
    print(model.fc)
    if dp.DEVICE == torch.device("cuda:0"):
        torch.cuda.empty_cache()
    # model.to(device=dp.DEVICE)
    # optim = torch.optim.Adam(params=model.parameters(), lr=LR)
    model = nn.DataParallel(model)
    criter = torch.nn.CrossEntropyLoss()
    train_ds, val_ds = dp.get_datasets()
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BS, shuffle=True, generator=torch.Generator(device=dp.DEVICE), num_workers=NUM_WORKERS)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BS, shuffle=False, generator=torch.Generator(device=dp.DEVICE), num_workers=NUM_WORKERS)

    # is_load = int(input("Load model? (0/1) "))
    is_load = 0
    if is_load:
        model.load_state_dict(torch.load(PATH))
    else:
        # model.load_state_dict(torch.load("saves/maxf1_87.pth"))
        train_losses, val_losses, train_f1, val_f1, train_acc, val_acc = training(model, criter, LRS, train_dl, val_dl, EPOCHES)
        torch.save(model.state_dict(), PATH)
        plot = metrics_plot(train_losses, val_losses, train_f1, val_f1, train_acc, val_acc)
        plot.savefig(f"plots/MetricsResNet50_{EPOCHES}Epoches.png")
        plot.show()
        # plt.show()

    data = val_ds[0]
    elem:torch.Tensor = data[0].to(dp.DEVICE)
    c, h, w = elem.shape
    elem = elem.reshape((1, c, h, w))
    label = torch.tensor(data[1])
    res = model(elem)
    print(res.shape, label.shape)
    loss = criter(res[0], label)
    print(loss)
