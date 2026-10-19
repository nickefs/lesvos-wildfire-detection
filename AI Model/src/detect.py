from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from pathlib import Path

# Load model
model = YOLO("models/best.pt")

# Find all images
images = list(Path("images/test").glob("*.jpg"))

# Run detection on CPU
for index,image in enumerate(images):
    results = model.predict(
        source=str(image),
        conf=0.25,
        iou=0.45,
        imgsz=960,
        device="cpu"
    )
    # Save annotated image
    results[0].save(filename=f"images/results/{image.stem}_result.jpg")

    # Print detected objects
    for box in results[0].boxes:
        cls = int(box.cls[0])
        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()

        print(model.names[cls], confidence, coordinates)

# Print detected objects
for box in results[0].boxes:
    cls = int(box.cls[0])
    confidence = float(box.conf[0])
    coordinates = box.xyxy[0].tolist()

    print(model.names[cls], confidence, coordinates)