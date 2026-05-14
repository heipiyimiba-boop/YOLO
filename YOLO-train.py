from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # 用更大模型，精度更高

model.train(
    data="./datas/coco128/coco128.yaml",
    epochs=100,        # 从30→100，最关键
    imgsz=640,
    batch=2,           # CPU友好
    device="cpu",
    augment=True,      # 开启数据增强
    conf=0.25,         # 降低置信度
    iou=0.45
)
