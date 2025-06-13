import os
import shutil

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
print(model)

# op = os.path

# dir = "images"
# train = "train"
# val = "val"
# imgdir = op.join(train, "images")
# labdir = op.join(train, "labels")
# imgs = sorted(os.listdir(imgdir))
# labs = sorted(os.listdir(labdir))

# valimg = op.join(val, "images")
# vallab = op.join(val, "labels")

# # if op.exists(valdir):
# #     shutil.rmtree(valdir)
# os.makedirs(valimg, exist_ok=True)
# os.makedirs(vallab, exist_ok=True)

# # for i in range(0, len(imgs), 1):
# #     shutil.move(op.join(maindir, imgs[i]), valdir)

# # print(len(os.listdir(maindir)))
# # print(len(os.listdir(valdir)))
# # ind = 0
# for img in imgs:
#     # if ind % 10 == 0:
#     #     ind += 1
#     #     continue
#     # ind += 1
#     name = img[:-4] + ".txt"
#     ind = labs.index(name)
#     # print(ind)
#     # shutil.move(op.join(imgdir, img), valimg)
#     # shutil.move(op.join(labdir, name), vallab)

# print(len(os.listdir(imgdir)))
# print(len(os.listdir(labdir)))
# print(len(os.listdir(valimg)))
# print(len(os.listdir(vallab)))
