from ultralytics import YOLO

print("\nStarting fast local training...\n")

model = YOLO("yolo11n.pt")

model.train(
    data="palm-tree-1/data.yaml",
    epochs=10,         # Minimum epochs for quick fine-tuning
    imgsz=416,         # Lower image size for much faster processing
    batch=16,          # Larger batch size if your GPU/CPU can handle it
    workers=2,         # Speeds up data loading
    device="cpu",     # Uses your hardware accelerator
    name="custom_palm_yolo11"
)

print("\n--- Training Finished! ---")


