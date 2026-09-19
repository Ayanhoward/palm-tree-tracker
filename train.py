from roboflow import Roboflow
from ultralytics import YOLO

# 1. Authenticate and download dataset (Roboflow handles the folder automatically)
rf = Roboflow(api_key="jFqWF9WGm7zMgcWqiALJ")
project = rf.workspace("capstone-p9zrm").project("palm-tree-jkpzn")
dataset = project.version(1).download("yolov8")

print("\nDataset downloaded successfully! Starting YOLO11 training...\n")

# 2. Train YOLO11 using the automatically generated data.yaml path
model = YOLO("yolo11n.pt")

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=40,
    imgsz=640,
    batch=8,
    name="custom_palm_yolo11"
)

print("\n--- Training Finished! ---")