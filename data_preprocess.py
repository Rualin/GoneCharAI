import os
import multiprocessing

import numpy as np
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.v2

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", True)

FLOWERS_NAMES = ["Daffodil", "Snowdrop", "LilyValley", "BlueBell", 
                "Crocus", "Iris", "TigerLily", "Tulip", "Fritillary",
                "Sunflower", "Daisy", "Colts`Foot", "Dandelion",
                "Cowslip", "Buttercup", "Windflower", "Pansy"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

def augment(image) -> list[np.ndarray]:
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # reversed = cv2.flip(image, 0)
    h, w, c = image.shape
    res = []
    rot_mat1 = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
    rot_mat2 = cv2.getRotationMatrix2D((w / 2, h / 2), -30, 1.0)
    res.append(cv2.warpAffine(image, rot_mat1, (w, h), flags=cv2.INTER_LINEAR))
    res.append(image)
    res.append(cv2.warpAffine(image, rot_mat2, (w, h), flags=cv2.INTER_LINEAR))
    # res.append(cv2.warpAffine(reversed, rot_mat1, (w, h), flags=cv2.INTER_LINEAR))
    # res.append(reversed)
    # res.append(cv2.warpAffine(reversed, rot_mat2, (w, h), flags=cv2.INTER_LINEAR))
    # res = [image,]
    return res

def partition_augmentation():
    maindir = "readyimgs"
    os.system("rm -rf " + maindir)
    os.mkdir(maindir)
    os.system("mkdir " + maindir + "/train")
    os.system("mkdir " + maindir + "/val")
    list_imgs = os.listdir("jpg")
    images = []
    for i in range(17):
        curr = []
        for j in range(80):
            if (j % 4) == 0:
                os.system("cp jpg/" + list_imgs[i * 80 + j] + " readyimgs/val")
            else:
                curr.append(list_imgs[i * 80 + j])
        images.append(curr)
    print(len(images))
    for el in images:
        print(len(el))
        # print(el)
    # print(images[0][0], images[16][0], images[16][-1])
    for i in range(17):
        for j in range(60):
            image = cv2.imread("jpg/" + images[i][j])
            # h, w, c = image.shape
            # top = (800 - h) // 2
            # bottom = 800 - h - top
            # left = (800 - w) // 2
            # right = 800 - w - left
            # image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
            sub = augment(image)
            for k in range(len(sub)):
                # cv2.imwrite("readyimgs\\train\\" + str(i * 60 + j) + "aug" + str(k) + ".jpg", sub[k])
                cv2.imwrite("readyimgs/train/" + images[i][j][:-4] + "aug" + str(k) + ".jpg", sub[k])


def onehot(num):
    res = torch.zeros(len(FLOWERS_NAMES), device=DEVICE)
    res[num] = 1
    return res

def preprocess(image, to_resize=False):
    preproc = torchvision.transforms.Compose([
    # torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.v2.Resize((550, 600))
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # plt.imshow(image)
    # plt.show()
    # print(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = preproc(image)
    if to_resize:
        res = torchvision.transforms.v2.Resize((550, 600))(res)
    # c, h, w = res.shape
    # res = res.reshape(1, c, h, w)
    return res

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, to_resize):
        super().__init__()
        self.path = path
        self.listdirr = os.listdir(path)
        self.to_resize = to_resize
    
    def __len__(self):
        return len(self.listdirr)

    def __getitem__(self, index) -> tuple[torch.Tensor,torch.Tensor]:
        oneclass = len(self.listdirr) // 17
        image = cv2.imread(self.path + "/" + self.listdirr[index])
        image = preprocess(image, self.to_resize)
        return image, (index // oneclass)

def get_datasets():
    '''
    Returns train, val datasets. Elements is tuples (image, label)
    '''
    # # result = []
    # images = []
    # for dir in list_imgs:
    #     img = cv2.imread("jpg/" + dir)
    #     images.append(img)
    #     # img = preprocess(img)
    #     # result.append(img)
    # labels = []
    # for i in range(17):
    #     sub = [i] * 80
    #     labels.extend(sub)
    # key = np.random.random(len(labels))
    # key = np.argsort(key)
    # res = [images[i] for i in key]
    # lab = [labels[i] for i in key]
    # leng = len(res)
    # fb = int(leng * 0.7)
    # sb = (leng - fb) // 2
    # for i in range(leng):
    #     res[i] = preprocess(res[i], to_resize=True)
    # for i in range(fb):
    #     res[i] = preprocess(res[i], to_resize=True)
    # for i in range(fb, leng):
    #     res[i] = preprocess(res[i], to_resize=False)
    return ImageDataset("readyimgs/train", True), ImageDataset("readyimgs/val", True)

if __name__ == "__main__":
    partition_augmentation()
