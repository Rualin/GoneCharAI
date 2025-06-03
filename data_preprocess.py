import os
import random

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import torchvision


DEVICE = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(DEVICE)
DATA_DIR = "tiff"

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


class RoadDataset(Dataset):

    """Massachusetts Road Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            class_rgb_values=class_rgb_values, 
            augmentation=None, 
            preprocessing=None,
        ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Lengths of image list and mask list are unequal")

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))
        image = torchvision.transforms.ToTensor()(image)
        image = image.to(dtype=torch.float32)
        # one-hot-encode the mask
        # mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        # print(mask.max(), mask.min())
        mask = mask / 255
        mask = torch.tensor(mask, dtype=torch.float32)
        # print(mask.shape)
        # print(mask.max(), mask.min())

        # # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']
        
        # # apply preprocessing
        # if self.preprocessing:
        #     # sample = self.preprocessing(image=image, mask=mask)
        #     # image, mask = sample['image'], sample['mask']
        #     image = self.preprocessing(image)
        #     mask = self.preprocessing(image)
        image = image.to(device=DEVICE)
        mask = mask.to(device=DEVICE)
        # print(image.shape, mask.shape)
        return image, mask
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)
    
# torchvision.transforms.Compose([
#     torchvision.transforms.
# ])

if __name__ == "__main__":
    dataset = RoadDataset(x_train_dir, y_train_dir, class_rgb_values=class_rgb_values)
    random_idx = random.randint(0, len(dataset) - 1)
    image, mask = dataset[random_idx]
    print(image.shape, mask.shape)

    visualize(
        original_image = torchvision.transforms.ToPILImage()(image),
        mask = mask.numpy(force=True).astype(np.int32)
    )
