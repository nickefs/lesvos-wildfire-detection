# Lesvos Wildfire Detection

Smoke and fire detection using Cloudinary and a custom Ultralytics YOLO model. This repository contains the AI component: CPU inference, annotated images, processing state, and monthly CSV logging.

## Structure

- `src/cloud_detect.py`: continuous Cloudinary detection and CSV logging.
- `src/detect.py`: local JPEG inference.
- `src/cloud_detect_v2.py`: empty placeholder.
- `models/best.pt`: separately supplied weights; excluded from Git.
- `images/` and `logs/`: local inputs, outputs, state, and logs; excluded.

Raspberry Pi capture/upload source was not found in the reviewed workspace. Supply that component separately.

## Setup

From the repository root in PowerShell:

```powershell
Set-Location 'AI Model'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On Linux, use `source .venv/bin/activate` and `cp .env.example .env`.

Fill in the three Cloudinary values in `.env` locally. Use an account permitted to search the relevant images. Never commit credentials. Place trusted, compatible weights at `models/best.pt`; no automatic download is implemented. Class names come from the weights. Model provenance, accuracy, training details, and redistribution rights were not available for review.

Dependency versions reflect the existing local environment; a fresh installation has not been validated.

## Cloud detection

Run from `AI Model`:

```powershell
python src/cloud_detect.py
```

The script loads its environment file and weights relative to its location. It polls Cloudinary every 60 seconds for the newest 20 images matching `asset_folder:Lesvos_Fire_Detection/* AND resource_type:image`. Camera uploads should use folders such as `Lesvos_Fire_Detection/CAMERA_01`.

Each batch is processed oldest first, skipping IDs in `images/cloud_results/processed.json`. Images are downloaded and evaluated on the CPU with confidence `0.25` and image size `960`. Annotated JPEGs are saved when detections exist. Processing state is updated, CSV records are appended, and downloaded originals are removed after successful logging. Errors are printed. Stop with Ctrl+C.

Monthly CSV files use asset creation dates: `logs/YYYY/detections_YYYY-MM.csv`.

```text
camera_id,timestamp,asset_id,detection,confidence,x1,y1,x2,y2,image_url
```

Each detection gets a row; images without detections get a `NONE` row. Camera ID is the final folder component, with an `UNKNOWN` fallback in CSV logging. Bounding boxes use pixel coordinates rounded to one decimal, and confidence is rounded to four decimals. Logs contain image URLs and should remain private.

## Local detection

Create `images/test` and `images/results` inside `AI Model`, add `.jpg` inputs, and run from `AI Model`:

```powershell
python src/detect.py
```

This script uses working-directory-relative paths, CPU inference, confidence `0.25`, IoU `0.45`, and image size `960`. It saves annotated images and prints labels, confidence, and coordinates. Its Hugging Face import is currently unused.

## Known limitations

- Search has no pagination; images outside the newest 20 can be missed.
- Processing state is saved before CSV logging; logging failure can leave missing records.
- State and CSV writes are not transactional or protected against concurrent workers. Run one worker.
- Processing state and annotated output have no retention policy.
- Console output directly accesses `asset_folder`; missing metadata can interrupt an iteration.
- Local detection repeats the last image's detections and errors when no inputs exist.
- The v2 script is empty. Alert delivery and verified camera deployment are not implemented in the available source.

These findings are documented without changing Python behavior. Detection performance and end-to-end operation were not tested in this review.

## Repository hygiene

Only AI detection source and setup/documentation files are included. Environment files, virtual environments, images, datasets, logs, state, and weights stay local. The root ignore rule excludes other workspace folders.

Review staged files and diffs before future commits. Ignore rules do not remove previously committed content from Git history. No project license has been selected; confirm ownership and model/dependency licensing before redistribution.
