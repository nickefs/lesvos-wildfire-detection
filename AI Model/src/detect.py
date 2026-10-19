from ultralytics import YOLO
from pathlib import Path

# Load model
BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "images" / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)
model = YOLO(str(BASE / "models" / "best.pt"))

# Find all images
images = list((BASE / "images" / "test").glob("*.jpg"))

# Run detection on CPU
for image in images:
    results = model.predict(
        source=str(image),
        conf=0.25,
        iou=0.45,
        imgsz=960,
        device="cpu"
    )
    # Save annotated image
    results[0].save(filename=str(OUTPUT / f"{image.stem}_result.jpg"))

    # Print detected objects
    for box in results[0].boxes:
        cls = int(box.cls[0])
        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()

        print(model.names[cls], confidence, coordinates)
