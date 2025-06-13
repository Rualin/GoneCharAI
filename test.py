import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


torch.set_default_device("cpu")

THRESHIOU = [(0.02, 0.5415724262628692), (0.05, 0.5752392123103611), 
             (0.1, 0.5943808204635416), (0.15, 0.6042670026833662), 
             (0.2, 0.6106475034734522), (0.3, 0.6189302407755173), 
             (0.4, 0.6240174568086888), (0.5, 0.6266541277417764), 
             (0.6, 0.6268637119437453), (0.7, 0.6237952999824858), 
             (0.8, 0.6142387542850246), (0.9, 0.5840248243283286), 
             (0.95, 0.5350216822773398), (0.98, 0.440629390463493), 
             (0.99, 0.36189994837784356)]
THRESHOLDS = [t[0] for t in THRESHIOU]
IOUS = np.array([t[1] for t in THRESHIOU])


plt.plot(THRESHOLDS, IOUS)

for xi, yi in zip(THRESHOLDS, IOUS):
    plt.annotate(
        f'({yi:.3f})',  # Текст подписи (можно настроить формат)
        (xi, yi),         # Координаты точки
        textcoords="offset points",  # Смещение относительно точки
        xytext=(-10, 10),   # Смещение текста (по вертикали)
        ha='center',       # Горизонтальное выравнивание
        fontsize=8        # Размер шрифта
    )

plt.savefig("thresh_iou.png")


# model = torchvision.models.segmentation.deeplabv3_resnet50(
#     weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
#     weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT
# )
# # print(model)
# x = torch.randn((1, 3, 256, 256))
# torch.onnx.export(model, x, f="torch_deeplab3.onnx", input_names=["features"], output_names=["logits"])
# model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
# model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
# # print("\n\n")
# print(model)

