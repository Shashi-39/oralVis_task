# Tooth Detection with YOLOv8

This project trains a YOLOv8 and YOLOv11 object detection model to identify teeth in intraoral dental images. The 32 target classes correspond to individual teeth as per the FDI World Dental Federation notation.

##  Dataset Structure

The dataset is organized in YOLO format with separate folders for images and labels.

```
oralVista/
├── split_data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── data.yaml
```

##  Label Format

Each label file contains YOLO annotations:
```
<class_id> <x_center> <y_center> <width> <height>
```

##  Environment Setup

Recommended Python environment (Python 3.10+):
```bash
conda create -n venv_cuda python=3.10 -y
conda activate venv_cuda
pip install ultralytics opencv-python pandas matplotlib seaborn scikit-learn
```

##  Training Command

We used the YOLOv11s model for training.

```python
from ultralytics import YOLO

WEIGHTS = "yolo11s.pt"
model = YOLO(WEIGHTS)

model.train(
    data="/content/drive/MyDrive/oralVista/split_data/data.yaml",
    imgsz=640,
    epochs=100,
    batch=16,
    device="cuda:0",
    workers=2,
)

 
```

## 📤 Inference on Validation & Test Sets

```python
from ultralytics import YOLO

model = YOLO("runs/detect/oralvista-v1/weights/best.pt")

pred_val = model.predict(
    source="/content/drive/MyDrive/oralVista/split_data/images/val",
    save_txt=True, save_conf=True, imgsz=640, conf=0.25, iou=0.7,
    project="oralVis/yolo_preds", name="val", 
)

pred_test = model.predict(
    source="/content/drive/MyDrive/oralVista/split_data/images/test",
    save_txt=True, save_conf=True, imgsz=640, conf=0.25, iou=0.7,
    project="oralVis/yolo_preds", name="test",  
)
```

## 📘 Notes

- FDI class names were embedded directly in `data.yaml` under the `names:` field.
- The dataset must be mounted via Google Drive or manually placed in the corresponding `/content/drive/MyDrive/oralVista/` path for training and evaluation.
