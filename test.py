import torch
import torchvision


torch.set_default_device("cpu")

model = torchvision.models.segmentation.deeplabv3_resnet50(
    weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
    weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT
)
# print(model)
x = torch.randn((1, 3, 256, 256))
torch.onnx.export(model, x, f="torch_deeplab3.onnx", input_names=["features"], output_names=["logits"])
model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
# print("\n\n")
print(model)

