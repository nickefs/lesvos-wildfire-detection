
from pathlib import Path
import os
import json
import time
import requests
import cloudinary
import cloudinary.api
from dotenv import load_dotenv
from ultralytics import YOLO
import csv

# Paths
BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "images" / "cloud_results"
OUTPUT.mkdir(parents=True, exist_ok=True)
STATE = OUTPUT / "processed.json"
PROJECT_FOLDER = "Lesvos_Fire_Detection"
# Cloudinary credentials
load_dotenv(BASE / ".env")

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"],
    secure=True
)

# Load YOLO
model = YOLO(str(BASE / "models" / "best.pt"))

# Remember which images we already processed
processed = set(
    json.loads(STATE.read_text(encoding="utf-8"))
) if STATE.exists() else set()

def save_csv(asset, result):

    # Date from Cloudinary
    timestamp = asset["created_at"]
    year = timestamp[:4]
    month = timestamp[:7]

    # Monthly CSV folder
    log_dir = BASE / "logs" / year
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / f"detections_{month}.csv"

    # Camera identification
    folder = asset.get("asset_folder") or asset.get("folder") or ""
    camera_id = folder.split("/")[-1] if folder else "UNKNOWN"

    # Check if CSV is new
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        if new_file:
            writer.writerow([
                "camera_id",
                "timestamp",
                "asset_id",
                "detection",
                "confidence",
                "x1", "y1", "x2", "y2",
                "image_url"
            ])

        # No detections
        if len(result.boxes) == 0:

            writer.writerow([
                camera_id,
                timestamp,
                asset["asset_id"],
                "NONE",
                "",
                "", "", "", "",
                asset["secure_url"]
            ])

        # One row for each detection
        else:

            for box in result.boxes:

                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                writer.writerow([
                    camera_id,
                    timestamp,
                    asset["asset_id"],
                    model.names[cls],
                    round(confidence, 4),
                    round(x1, 1),
                    round(y1, 1),
                    round(x2, 1),
                    round(y2, 1),
                    asset["secure_url"]
                ])

while True:
    try:
        # Get the 20 most recent Cloudinary images
        response = (
          cloudinary.Search()
         .expression(
           f"asset_folder:{PROJECT_FOLDER}/* AND resource_type:image"
         )
         .sort_by("created_at", "desc")
         .max_results(20)
        .execute()
        )
     
        # Process oldest first
        for asset in reversed(response["resources"]):
            asset_id = asset["asset_id"]

            if asset_id in processed:
                continue

            # Download original image
            image_path = OUTPUT / f"{asset_id}.{asset['format']}"

            download = requests.get(
                asset["secure_url"],
                timeout=60
            )
            download.raise_for_status()
            image_path.write_bytes(download.content)

            # Run YOLO inference
            result = model.predict(
                source=str(image_path),
                conf=0.25,
                imgsz=960,
                device="cpu",
                verbose=False
            )[0]

            # Save annotated image
            result_path = OUTPUT / f"{asset_id}_result.jpg"
            if len(result.boxes) > 0:
              result.save(filename=str(result_path))

            # Print detections
            print(f"\nProcessed: {asset['public_id']}")
            print("Camera folder:", asset["asset_folder"])

            for box in result.boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                print(model.names[cls], round(confidence, 3))

            # Mark as processed only after successful inference
            processed.add(asset_id)
            STATE.write_text(
                json.dumps(sorted(processed), indent=2),
                encoding="utf-8"
            )
            # Save YOLO results to monthly CSV
            save_csv(asset, result)
            # Delete downloaded image after successful processing
            image_path.unlink(missing_ok=True)

    except Exception as error:
        print("Error:", error)

    # Check Cloudinary every 60 seconds
    print("Sleeping for 60 seconds before checking for new images...")
    time.sleep(60)
