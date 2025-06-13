import os
import shutil

import numpy as np
import cv2
from tqdm import tqdm


DATA_DIR = "tiff"
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')
x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_labels')
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')
RES_DIR = "croped"

def copy_crop(x_dir, y_dir, cnt=3):
    x_ims = os.listdir(os.path.join(DATA_DIR, x_dir))
    y_ims = os.listdir(os.path.join(DATA_DIR, y_dir))
    if len(x_ims) != len(y_ims):
        raise Exception("Lengths of dirs are unequal!!!")

    if not os.path.exists(RES_DIR):
        os.mkdir(RES_DIR)
    if not os.path.exists(os.path.join(RES_DIR, x_dir)):
        os.mkdir(os.path.join(RES_DIR, x_dir))
    if not os.path.exists(os.path.join(RES_DIR, y_dir)):
        os.mkdir(os.path.join(RES_DIR, y_dir))

    step = 1500 // cnt
    bound = step * (cnt - 1) + 1
    for i in tqdm(range(len(x_ims))):
        x_img = cv2.imread(os.path.join(DATA_DIR, x_dir, x_ims[i]))
        y_img = cv2.imread(os.path.join(DATA_DIR, y_dir, y_ims[i]))

        for x in range(0, bound, step):
            for y in range(0, bound, step):
                x_crop = x_img[x:(x + step), y:(y + step), :]
                # start counting of white pixels

                white_pixels = np.all(x_crop == [255, 255, 255], axis=-1)
                white_ratio = np.sum(white_pixels) / white_pixels.size
                if white_ratio > 0.3:
                    continue

                # end counting white pixels
                y_crop = y_img[x:(x + step), y:(y + step), :]
                if (y_crop.mean() == 0):
                    continue

                x_path = os.path.join(RES_DIR, x_dir, x_ims[i][:(-5)])
                y_path = os.path.join(RES_DIR, y_dir, y_ims[i][:(-4)])
                cv2.imwrite(f"{x_path}_{x // step}_{y // step}.png", x_crop)
                cv2.imwrite(f"{y_path}_{x // step}_{y // step}.png", y_crop)


if __name__ == "__main__":
    if (os.path.exists(RES_DIR)):
        shutil.rmtree(RES_DIR)
    os.mkdir(RES_DIR)

    copy_crop("train", "train_labels")
    # copy_crop("val", "val_labels")
