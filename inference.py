import os

import torch
import torchvision

import data_preprocess as dp
import loops as ls
import useful_funcs as fuc

# THRESHOLDS = [0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
THRESHOLDS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
# THRESHOLDS = [0.99, 0.999, 0.9999, 0.99999, 0.999999]
CRITER = fuc.BCEDICELoss()

if __name__ == "__main__":
    test_ds = dp.RoadDataset(dp.x_test_dir, dp.y_test_dir,
                             transform=dp.VALID_TRANSFORM, bs=1)
    test_dl = torch.utils.data.DataLoader(test_ds, 1, generator=torch.Generator(device=dp.DEVICE))

    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
        weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT
    )
    model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    # model = torch.nn.DataParallel(model)
    model.to(dp.DEVICE)
    model.load_state_dict(torch.load(os.path.join("saves", "maxIoU_64.pth")))
    ious = []

    for threshold in THRESHOLDS:
        print("Threshold:", threshold)

        test_loss, test_conf = ls.test_loop(model, CRITER, test_dl, threshold)
        test_metrics = fuc.calculate_metrics(test_conf)
        print("Test loss:", test_loss)
        for key in test_metrics:
            print(f"{key}: {test_metrics[key]}")
        ious.append(test_metrics["IoU"])
    
    print(THRESHOLDS)
    print(ious)
    print(list(zip(THRESHOLDS, ious)))
