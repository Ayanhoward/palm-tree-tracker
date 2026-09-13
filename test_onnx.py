from ultralytics import YOLO
import cv2

# Load the newly generated ONNX model
model = YOLO("best.onnx")

# Run inference on your input video
results = model.track(
    source="input_video.mp4",
    show=False,
    save=True,
    conf=0.25
)

print("ONNX tracking completed. Check the 'runs/detect/' folder for the output video.")