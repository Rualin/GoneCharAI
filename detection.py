import os

from ultralytics import YOLO


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")

    model.train(data="data.yaml", epochs=100, 
                          batch=16, workers=1,
                          copy_paste=0.5, flipud=0.5,
                          perspective=0.00005, scale=0.3,
                          degrees=45)
