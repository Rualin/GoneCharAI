# GoneCharAI

## Detection lab work

---

### About

Program to detect peoples from [Heridal Computer Vision Project](https://universe.roboflow.com/new-workspace-qsuag/heridal-t428z "https://universe.roboflow.com/new-workspace-qsuag/heridal-t428z") (also on [Kaggle](https://www.kaggle.com/datasets/imadeddinelassakeur/heridal/data "https://www.kaggle.com/datasets/imadeddinelassakeur/heridal/data")) \
Dataset must be in YOLOv8 format \
\
I achieved 81% mAP50 in 100 epoche. \
\
Thoughts to optimize code for that task:
- Switch to YOLO11
- Remove large ([256, 20, 20]) and middle ([128, 40, 40]) scales
- Distill this model
- Switch to float16
- Decrease coefficient for bbox loss and increase for dfloss
- Think about the usefulness of empty crops: are they useful or harmful? On average, they are useful, but in this task, almost the entire picture is the background, so are they needed?
- Train on larger crops 

These notes can increase Pareto efficiency: work speed/accuracy ratio

---

### Starting

```
pip install -r requirements
```
```
python detection.py
```