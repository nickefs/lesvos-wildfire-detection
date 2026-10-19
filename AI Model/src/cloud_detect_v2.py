
from pathlib import Path
import csv
import json
import os
import time

import cv2
import cloudinary
import requests
from dotenv import load_dotenv
from ultralytics import YOLO


# ========== SETTINGS ==========

BASE = Path(__file__).resolve().parent.parent

OUTPUT = BASE / "images" / "cloud_results"
OUTPUT.mkdir(parents=True, exist_ok=True)

STATE_FILE = OUTPUT / "processed_v2.json"
CSV_FILE = OUTPUT / "detections.csv"

PROJECT_FOLDER = "Lesvos_Fire_Detection"

CHECK_INTERVAL = 30
PAGE_SIZE = 100
MAX_PAGES = 3

# Initially, run YOLO on ALL images
ENABLE_CHANGE_GATE = False

# Experimental change thresholds
GLOBAL_TRIGGER_PCT = 0.5
LOCAL_TRIGGER_PCT = 1.0
MAX_SKIPS = 2

CSV_FIELDS = [
    "camera",
    "created_at",
    "public_id",
    "asset_id",
    "change_global_pct",
    "change_local_pct",
    "yolo_ran",
    "reason",
    "detection",
    "confidence",
    "x1", "y1", "x2", "y2",
    "image_url"
]


# ========== CLOUDINARY AND YOLO ==========

load_dotenv(BASE / ".env")

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"],
    secure=True
)

model = YOLO(str(BASE / "models" / "best.pt"))

if STATE_FILE.exists():
    processed = set(json.loads(STATE_FILE.read_text(encoding="utf-8")))
else:
    processed = set()

# Previous frame for each camera
previous_frames = {}

# Number of consecutive skipped frames per camera
skipped_frames = {}


# ========== IMAGE COMPARISON ==========

def compare_frames(previous, image):

    # Convert current image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize only for change detection
    gray = cv2.resize(gray, (1280, 720))

    if previous is None:
        return gray, None, None

    # Absolute pixel differences
    difference = cv2.absdiff(previous, gray)

    # Pixels whose brightness changed by more than 20
    changed = difference > 20

    # Percentage of changed pixels in whole image
    global_pct = float(changed.mean() * 100)

    # Divide image into 40x40 pixel regions
    tiles = changed.reshape(18, 40, 32, 40)

    # Find region with highest percentage of change
    local_pct = float(
        tiles.mean(axis=(1, 3)).max() * 100
    )

    return gray, global_pct, local_pct


# ========== GET CLOUDINARY IMAGES ==========

def fetch_recent_assets():

    search = (
        cloudinary.Search()
        .expression(
            f"asset_folder:{PROJECT_FOLDER}/* AND resource_type:image"
        )
        .sort_by("created_at", "desc")
        .max_results(PAGE_SIZE)
    )

    assets = []
    cursor = None

    for _ in range(MAX_PAGES):

        if cursor:
            search.next_cursor(cursor)

        response = search.execute()

        assets.extend(response.get("resources", []))

        cursor = response.get("next_cursor")

        if not cursor:
            break

    # Process from oldest to newest
    return sorted(
        assets,
        key=lambda a: (a["created_at"], a["asset_id"])
    )


# ========== SAVE CSV ==========

def write_rows(rows):

    new_file = (
        not CSV_FILE.exists()
        or CSV_FILE.stat().st_size == 0
    )

    with CSV_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=CSV_FIELDS
        )

        if new_file:
            writer.writeheader()

        writer.writerows(rows)


# ========== PROCESS ONE IMAGE ==========

def process_asset(asset):

    asset_id = asset["asset_id"]

    # Identify camera from Cloudinary folder
    folder = asset.get("asset_folder") or asset.get("folder") or ""
    camera = folder.rsplit("/", 1)[-1] if folder else "Unknown_Camera"

    image_path = OUTPUT / f"{asset_id}.jpg"

    try:
        # Download image
        response = requests.get(
            asset["secure_url"],
            timeout=60
        )

        response.raise_for_status()
        image_path.write_bytes(response.content)

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Cannot read image")

        # Compare with previous image from SAME camera
        gray, global_pct, local_pct = compare_frames(
            previous_frames.get(camera),
            image
        )

        # Decide whether YOLO should run
        if camera not in previous_frames:

            run_yolo = True
            reason = "first_image"

        elif not ENABLE_CHANGE_GATE:

            run_yolo = True
            reason = "measure_only"

        elif (
            global_pct >= GLOBAL_TRIGGER_PCT
            or local_pct >= LOCAL_TRIGGER_PCT
        ):

            run_yolo = True
            reason = "change_detected"

        elif skipped_frames.get(camera, 0) >= MAX_SKIPS:

            run_yolo = True
            reason = "periodic_check"

        else:

            run_yolo = False
            reason = "small_change"

        # Common data for CSV
        base_row = {
            "camera": camera,
            "created_at": asset["created_at"],
            "public_id": asset["public_id"],
            "asset_id": asset_id,
            "change_global_pct": (
                "" if global_pct is None else round(global_pct, 4)
            ),
            "change_local_pct": (
                "" if local_pct is None else round(local_pct, 4)
            ),
            "yolo_ran": run_yolo,
            "reason": reason,
            "detection": "",
            "confidence": "",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "image_url": asset["secure_url"]
        }

        rows = []

        # ========== YOLO ==========

        if run_yolo:

            result = model.predict(
                source=str(image_path),
                conf=0.25,
                iou=0.45,
                imgsz=960,
                device="cpu",
                verbose=False
            )[0]

            if len(result.boxes) == 0:

                rows.append({
                    **base_row,
                    "detection": "NONE"
                })

            else:

                for box in result.boxes:

                    x1, y1, x2, y2 = map(
                        float,
                        box.xyxy[0].tolist()
                    )

                    rows.append({
                        **base_row,
                        "detection": model.names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]), 4),
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1)
                    })

                # Save annotated image
                result.save(
                    filename=str(
                        OUTPUT / f"{asset_id}_result.jpg"
                    )
                )

            skipped_frames[camera] = 0

        else:

            rows.append({
                **base_row,
                "detection": "SKIPPED"
            })

            skipped_frames[camera] = (
                skipped_frames.get(camera, 0) + 1
            )

        # Save detection results
        write_rows(rows)

        # Update previous image for this camera
        previous_frames[camera] = gray

        # Remember processed image
        processed.add(asset_id)

        STATE_FILE.write_text(
            json.dumps(sorted(processed), indent=2),
            encoding="utf-8"
        )

        print(
            f"{camera} | "
            f"Global: {global_pct} | "
            f"Local: {local_pct} | "
            f"{reason} | "
            f"{[r['detection'] for r in rows]}"
        )

    finally:

        # Delete temporary downloaded image
        image_path.unlink(missing_ok=True)


# ========== MAIN LOOP ==========

print("Cloud Fire Detection v2 started")

try:

    while True:

        try:

            assets = fetch_recent_assets()

            for asset in assets:

                if asset["asset_id"] not in processed:

                    try:
                        process_asset(asset)

                    except Exception as error:
                        print("Image error:", error)

        except Exception as error:
            print("Cloudinary error:", error)

        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:

    print("Program stopped")
