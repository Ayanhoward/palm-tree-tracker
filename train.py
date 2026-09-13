import os
from roboflow import Roboflow
from ultralytics import YOLO

# 1. Authenticate with Roboflow
rf = Roboflow(api_key="jFqWF9WGm7zMgcWqiALJ")

print("Downloading dataset...")
project = rf.workspace("capstone-p9zrm").project("palm-tree-jkpzn")
version = project.version(1)

# Download directly into a clean local folder named "dataset"
dataset = version.download("yolov8", location="dataset")

print("\nDataset ready! Starting YOLOv8 training...\n")

# 2. Train YOLOv8 locally
model = YOLO("yolov8n.pt")

model.train(
    data=os.path.join(dataset.location, "data.yaml"),
    epochs=40,
    imgsz=640,
    batch=8,
    name="custom_palm_model"
)

print("\n--- Training Finished! ---")
print("Your trained model is saved at: runs/detect/custom_palm_model/weights/best.pt")