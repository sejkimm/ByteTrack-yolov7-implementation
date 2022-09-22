import os
import sys

from yolov7.detector import YOLOv7Detector

sys.path.insert(0, "./yolov7")

detector = YOLOv7Detector(
    device="cpu",
    weight_dir=os.getcwd() + "/yolov7/model/yolov7.pt",
    image_size=640,
    confidence_threshold=0.2,
    iou_threshold=0.45,
    classes=None,
    agnostic_nms=False,
)