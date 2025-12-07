from ultralytics import YOLO

# 1. Load YOLO model (small & fast)
model = YOLO("yolov8n.pt")   # you can use yolov8s.pt also

# 2. Train the model
model.train(
    data="config.yaml",     # path to your YAML file
    epochs=20,            # increase if you want better accuracy
    imgsz=256,            # image size
    batch=2,              # small batch for CPU
    device="cpu",         # training on CPU
    workers=1             # important for Windows
)

print("Training completed successfully!")