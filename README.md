# Lesvos Wildfire Detection

Raspberry Pi camera capture and Cloudinary upload, with CPU smoke/fire detection using a custom Ultralytics YOLO model.

## Structure

```text
AI Model/
  .env.example
  src/
    detect.py
    cloud_detect.py
    cloud_detect_v2.py
  images/       # local inputs and results
  logs/         # local monthly CSV logs
  models/       # supply best.pt locally
Raspberry Pi/
  camera_upload.py
requirements.txt
README.md
```

Empty runtime folders are preserved with `.gitkeep`. Credentials, virtual environments, images, logs, state, and weights remain local.

## AI setup

From the repository root with Python 3.11:

```powershell
python -m venv "AI Model/.venv"
& "./AI Model/.venv/Scripts/Activate.ps1"
python -m pip install -r requirements.txt
Copy-Item "AI Model/.env.example" "AI Model/.env"
```

On Linux, activate with `source "AI Model/.venv/bin/activate"` and copy the template with `cp "AI Model/.env.example" "AI Model/.env"`.

Fill in Cloudinary credentials and place trusted, compatible weights at `AI Model/models/best.pt`. No automatic model download is implemented. Dependency pins came from the existing project; a fresh installation has not been verified.

Run one cloud detector at a time:

```powershell
python "AI Model/src/cloud_detect.py"
python "AI Model/src/cloud_detect_v2.py"
```

The original detector polls the newest 20 images every 60 seconds, saves monthly CSV files under `AI Model/logs/YYYY/`, and tracks processed IDs in `images/cloud_results/processed.json`.

The experimental v2 detector polls every 30 seconds and fetches up to three pages of 100 images. It measures frame changes per camera; its optional change gate is disabled by default, so every unprocessed image runs through YOLO. It writes `images/cloud_results/detections.csv` and uses separate state in `processed_v2.json`. Switching to v2 reprocesses images seen by the original detector.

Both search Cloudinary folders under `Lesvos_Fire_Detection/`, use CPU inference, and save annotated images when detections exist. Stop with Ctrl+C.

For local detection, put JPEGs in `AI Model/images/test/` and run:

```powershell
python "AI Model/src/detect.py"
```

Results go to `AI Model/images/results/`. Paths resolve relative to the script.

## Raspberry Pi

Install `rpicam-still` and `curl`. The script requires a supported camera and uploads through `wlan0`. Set environment variables in the shell; this script does not load `.env`:

```bash
export CLOUDINARY_CLOUD_NAME='your-cloud-name'
export CLOUDINARY_UPLOAD_PRESET='your-unsigned-upload-preset'
export CLOUDINARY_ASSET_FOLDER='Lesvos_Fire_Detection/Camera_001'
python3 'Raspberry Pi/camera_upload.py'
```

Configure the unsigned upload preset in Cloudinary and use a distinct folder per camera. The script captures a temporary JPEG, uploads it, deletes it after success, and waits 60 seconds. Capture/upload failures stop the script; failed uploads can leave temporary photos. Camera focus options depend on hardware support.

## Limitations

Search limits can miss a backlog. CSV and state writes are not transactional, so crashes can produce duplicate records. Run one worker at a time. Runtime data has no retention policy. V2 frame comparison is experimental. Detection accuracy and end-to-end camera/cloud operation have not been validated. Alert delivery is not implemented.

Keep credentials and image URLs private. Model provenance and redistribution rights must be established separately. No project license has been selected.
