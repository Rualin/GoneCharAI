import os
import numpy as np
from PIL import Image
from typing import Tuple

from tqdm import tqdm

def split_image(img: np.ndarray, size: Tuple[int, int]) -> list:

    height, width = img.shape[0], img.shape[1]
    tile_h, tile_w = size
    n_tiles_h = height // tile_h
    n_tiles_w = width // tile_w
    tiles = []
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            y_start = i * tile_h
            y_end = y_start + tile_h
            x_start = j * tile_w
            x_end = x_start + tile_w
            tile = img[y_start:y_end, x_start:x_end]
            tiles.append((tile, (i, j)))
    return tiles

def cropped_data(imgs_dir: str, masks_dir: str, size: Tuple[int, int]) -> None:

    imgs_crop_dir = imgs_dir.rstrip('/\\') + '_crop'
    masks_crop_dir = masks_dir.rstrip('/\\') + '_crop'
    os.makedirs(imgs_crop_dir, exist_ok=True)
    os.makedirs(masks_crop_dir, exist_ok=True)

    # Поддержка tif и tiff
    valid_img_exts = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
    valid_mask_exts = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')

    x_files = sorted([f for f in os.listdir(imgs_dir) if f.lower().endswith(valid_img_exts)])
    y_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(valid_mask_exts)])

    if len(x_files) != len(y_files):
        raise ValueError(f"Количество файлов не совпадает: {len(x_files)} vs {len(y_files)}")

    for img_name, mask_name in tqdm(zip(x_files, y_files)):
        img_path = os.path.join(imgs_dir, img_name)
        mask_path = os.path.join(masks_dir, mask_name)

        with Image.open(img_path) as img, Image.open(mask_path) as mask:
            img_arr = np.array(img)
            mask_arr = np.array(mask)
            if img_arr.shape[:2] != mask_arr.shape[:2]:
                print(f"Размеры не совпадают для {img_name} и {mask_name} - пропускаем")
                continue
            img_tiles = split_image(img_arr, size)
            mask_tiles = split_image(mask_arr, size)
            base_name = os.path.splitext(img_name)[0]

            for (img_tile, _), (mask_tile, (i, j)) in zip(img_tiles, mask_tiles):
                tile_name = f"{base_name}_{i}_{j}.png"
                
                white_pixels = np.all(img_tile == [255, 255, 255], axis=-1)
                white_ratio = np.sum(white_pixels) / white_pixels.size
                if white_ratio > 0.1:
                    continue

                Image.fromarray(img_tile).save(os.path.join(imgs_crop_dir, tile_name))
                Image.fromarray(mask_tile).save(os.path.join(masks_crop_dir, tile_name))

if __name__ == '__main__':
    TRAIN_IMGS = os.path.join("tiff", "train")
    TRAIN_MASKS = os.path.join("tiff", "train_labels")
    cropped_data(TRAIN_IMGS, TRAIN_MASKS, (500, 500))
