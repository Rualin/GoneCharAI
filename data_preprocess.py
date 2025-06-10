import os
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import albumentations as album


DEVICE = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(DEVICE)
DATA_DIR = "croped2"
TEST = True

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')


class_dict = pd.read_csv("label_class_dict.csv")
# Get class names
class_names = class_dict['name'].tolist()
# Get class RGB values
class_rgb_values = class_dict[['r','g','b']].values.tolist()

print('All dataset classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)


# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(18,6))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

TRAIN_CROP_SIZE = 256

TRAIN_TRANSFORM = album.Compose([
        # Базовые аугментации
        album.RandomCrop(height=TRAIN_CROP_SIZE, width=TRAIN_CROP_SIZE, p = 1),
        
        # Цветовые аугментации
        album.OneOf([
            album.RandomGamma(gamma_limit=(80, 120), p=0.5),
            album.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                brightness_by_max=True,
                p=0.5
            ),
            album.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            album.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.5
            )
        ], p=0.75),
        
        # Геометрические аугментации
        album.OneOf([
            album.HorizontalFlip(p=0.5),
            album.VerticalFlip(p=0.5),
            album.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            )
        ], p=0.75),
        
        # Размытия и шумы
        album.OneOf([
            album.GaussianBlur(blur_limit=(3, 5), p=0.5),
            album.GaussNoise(p=0.5),
            album.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=0.5
            )
        ], p=0.5),
        album.ToTensorV2()
    ]
)
VALID_TRANSFORM = album.Compose([
    album.PadIfNeeded(min_height=1536, min_width=1536, border_mode=cv2.BORDER_CONSTANT),
    album.ToTensorV2()
])

class RoadDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            transform=None,
            bs=8
    ):
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        length = len(self.image_paths)
        if length % bs != 0:
            length = (length // bs) * bs
        self.image_paths = self.image_paths[:length]
        self.mask_paths = self.mask_paths[:length]

        if TEST and bs != 1:
            length = int((length / bs) * 0.2) * bs
            self.image_paths = self.image_paths[:length]
            self.mask_paths = self.mask_paths[:length]

        self.transform = transform

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
        # print(image.shape, mask.shape)
        # print("Image before:", image.mean(), image.std())
        # print("Mask before:", mask.mean(), mask.std())
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)

        image = augmented['image'].float()
        mask = augmented['mask'].float()
        # print("Image after:", image.mean(), image.std())
        # print("Mask after:", mask.mean(), mask.std())
        # print(image.shape, mask.shape)
        return image, mask.unsqueeze(0) 

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    dataset = RoadDataset(x_train_dir, y_train_dir)
    random_idx = random.randint(0, len(dataset) - 1)
    image, mask = dataset[random_idx]
    print(image.shape, mask.shape)

    visualize(
        original_image = torchvision.transforms.ToPILImage()(image),
        mask = mask.numpy(force=True).astype(np.int32)
    )
