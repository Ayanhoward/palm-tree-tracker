import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# 1. Load custom trained YOLOv8 model
model = YOLO("best.pt")

# 2. Load input video
video_path = "input_video (1).mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), f"Error: {video_path} not found in project directory."

w, h, fps = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    int(cap.get(cv2.CAP_PROP_FPS)),
)

video_writer = cv2.VideoWriter("counted_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

cv2.namedWindow("Palm Tree Counter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Palm Tree Counter", 1280, 720)

unique_palm_ids = set()

# Initialize Supervision Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

print("\n--- Running High-Sensitivity Smooth Palm Counter ---")

# 3. Stream generator mode handles memory & batching efficiently
results_generator = model.track(
    source=video_path,
    stream=True,         # Generator streaming prevents pipeline lag
    persist=True,        # Retains track IDs across frames
    tracker="bytetrack.yaml",
    imgsz=1280,         # Keeps full resolution for small/distant trees
    conf=0.01,          # High sensitivity for faint background palms
    iou=0.4,            # Handles overlapping fronds
    verbose=False
)

for result in results_generator:
    frame = result.orig_img.copy()

    # Convert results to Supervision format
    detections = sv.Detections.from_ultralytics(result)

    # Track unique palm IDs across frames
    if detections.tracker_id is not None:
        for tracker_id in detections.tracker_id:
            unique_palm_ids.add(int(tracker_id))

        labels = [
            f"Palm #{tracker_id}"
            for tracker_id in detections.tracker_id
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # Render Total Palms Count Overlay
    total_count = len(unique_palm_ids)
    cv2.rectangle(frame, (20, 20), (420, 80), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Total Palms Counted: {total_count}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    video_writer.write(frame)
    cv2.imshow("Palm Tree Counter", frame)

    # Smooth frame pacing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("========================================")
print(f" FINAL ACCURATE COUNT: {len(unique_palm_ids)} unique palm tree(s) tracked.")
print("========================================\n")