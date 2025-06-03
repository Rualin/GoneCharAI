import os
from math import ceil

import torch
import torchvision
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import data_preprocess as dp


PATH = os.path.join("saves", "save.pth")
BS = 2
EPOCHES = 100
LR = 1e-3
LRS = [1e-2, 1e-2, 1e-3, 1e-3, 1e-4, 1e-4, 1e-5]
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "sigmoid"
torch.set_default_device(dp.DEVICE)
loss = torch.nn.BCELoss()
METRICS = [
    smp.metrics.iou_score,
    smp.metrics.f1_score,
    smp.metrics.accuracy,
    smp.metrics.recall,
    smp.metrics.precision,
]
METRICS_NAMES = ["IoU", "F1Score", "Accuracy", "Recall", "Precision"]

def train_val_loop(model, criterion, optimizer, dl, is_val):
    if is_val:
        model.eval()
    else:
        model.train()
    running_loss = 0.0
    met_vals = torch.zeros(len(METRICS), dtype=torch.float32)
    for data in tqdm(dl):
        inputs = data[0].to(dp.DEVICE)
        labels = data[1].to(dp.DEVICE)
        if not(is_val):
            optimizer.zero_grad()
        outputs:torch.Tensor = model(inputs)
        outputs = outputs.view_as(labels)
        loss = criterion(outputs, labels)
        if not(is_val):
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        preds = torch.threshold(outputs, 0.5, 0)
        preds = preds.to(dtype=torch.bool)
        labels = labels.to(dtype=torch.bool)
        tp = (preds & labels).sum()
        tn = (~preds & ~labels).sum()
        fp = (preds & ~labels).sum()
        fn = (~preds & labels).sum()
        for i, metric in enumerate(METRICS):
            met = metric(tp, fp, fn, tn)
            # print(met.shape, met_vals.shape)
            # print(met.sum(dim=0))
            # print(met)
            met_vals[i] += met
    met_vals /= len(dl)
    if is_val:
        print("\tValid:")
    else:
        print("\tTrain:")
    print(f'\t\tLoss: {(running_loss / len(dl)):.3f}', end = "")
    for i in range(len(METRICS)):
        print(f", {METRICS_NAMES[i]}: {met_vals[i]}", end="")
    print()
    return running_loss / len(dl), met_vals


def metrics_plot(**metrics):
    length = len(metrics)
    ncols = 2
    nrows = ceil(length / 2)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for i, (name, val) in enumerate(metrics.items()):
        ax[i // ncols, i % ncols].plot(val)
        ax[i // ncols, i % ncols].set_title(name)
    fig.suptitle("Metrics")
    return fig


def training(model, train_dl, val_dl, criterion, lrs, epoches=10):
    train_losses = []
    val_losses = []
    val_metrics = []
    maxiou = 0
    lr_id = 0
    for epoch in range(epoches):
        if (epoch % 10 == 0):
            optimizer = torch.optim.Adam(model.parameters(), lr=lrs[lr_id])
            lr_id = min(len(lrs) - 1, lr_id + 1)
        print("Epoch:", epoch)
        print("LR:", lrs[lr_id])
        tr_loss, train_mets = train_val_loop(model, criterion, optimizer, train_dl, False)
        train_losses.append(tr_loss)
        val_loss, val_mets = train_val_loop(model, criterion, optimizer, val_dl, True)
        val_losses.append(val_loss)
        val_metrics.append(val_mets)
        if val_mets[0] > maxiou:
            if maxiou != 0:
                os.system("rm -f " + os.path.join("saves", f"maxIoU_{(maxiou * 100):.0f}.pth"))
            maxiou = val_mets[0]
            torch.save(model.state_dict(), os.path.join("saves", f"maxIoU_{(val_mets[0] * 100):.0f}.pth"))
        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join("saves", f"epoch_{epoch}.pth"))
            metrics = dict(zip(METRICS_NAMES + ["Train_loss", "Val_loss"], val_mets.tolist() + [train_losses, val_losses]))
            plot = metrics_plot(**metrics)
            if not(os.path.exists("plots")):
                os.mkdir("plots")
            plot.savefig(os.path.join("plots", f"MetricsDeepLabv3_{epoch}Epoches.png"))
        if tr_loss < 0.005:
            cnt_small += 1
        else:
            cnt_small = 0
        if cnt_small == 3:
            # torch.save(model.state_dict(), os.path.join("saves", f"epoch_{epoch}_small.pth"))
            # plot = metrics_plot(train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs)
            metrics = dict(zip(METRICS_NAMES + ["Train_loss", "Val_loss"], val_mets.tolist() + [train_losses, val_losses]))
            plot = metrics_plot(**metrics)
            if not(os.path.exists("plots")):
                os.mkdir("plots")
            plot.savefig(os.path.join("plots", f"MetricsDeepLabv3_{epoch}Epoches_small.png"))
            break
    return train_losses, val_losses, train_mets, val_mets

if __name__ == "__main__":
    train_ds = dp.RoadDataset(dp.x_train_dir, dp.y_train_dir, preprocessing=torchvision.transforms.Resize((1504, 1504)))
    val_ds = dp.RoadDataset(dp.x_valid_dir, dp.y_valid_dir, preprocessing=torchvision.transforms.Resize((1504, 1504)))
    test_ds = dp.RoadDataset(dp.x_test_dir, dp.y_test_dir, preprocessing=torchvision.transforms.Resize((1504, 1504)))
    train_dl = DataLoader(train_ds, BS, shuffle=True, generator=torch.Generator(device=dp.DEVICE))
    val_dl = DataLoader(val_ds, BS, generator=torch.Generator(device=dp.DEVICE))
    test_dl = DataLoader(test_ds, 1, generator=torch.Generator(device=dp.DEVICE))

    if dp.DEVICE == torch.device("cuda:1"):
        torch.cuda.empty_cache()

    model = smp.DeepLabV3(encoder_name=ENCODER,
                          encoder_weights=ENCODER_WEIGHTS,
                          encoder_output_stride=8,
                          classes=1,
                          activation=ACTIVATION)
    model.to(dp.DEVICE)

    batch = next(iter(train_dl))
    # print(type(batch))
    # print(len(batch))
    # print(batch[0].shape, batch[1].shape)
    res = model(batch[0])
    # print(res.shape)
    # print(batch.shape)

    # is_load = int(input("Load model? (0/1) "))
    is_load = 0
    if is_load:
        model.load_state_dict(torch.load(PATH))
    else:
        train_losses, val_losses, train_mets, val_mets = training(model, train_dl, val_dl, criterion=loss, lrs=LRS, epoches=EPOCHES)
        torch.save(model.state_dict(), PATH)
        metrics = dict(zip(METRICS_NAMES + ["Train_loss", "Val_loss"], val_mets.tolist() + [train_losses, val_losses]))
        plot = metrics_plot(**metrics)
        plot.savefig(os.path.join("plots", f"MetricsDeepLabv3_{EPOCHES}Epoches.png"))
        plot.show()
