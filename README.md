# Lesvos Fire Detection

Camera-based smoke and fire detection for Lesvos, using Cloudinary image storage and a custom Ultralytics YOLO model. The available implementation performs CPU inference, saves annotated detections, and writes monthly CSV records. This workspace also contains QGIS scripts for planning camera locations.

## Current implementation

```text
Raspberry Pi camera -> Cloudinary -> YOLO inference -> CSV + annotated images
  not present here     external      cloud_detect.py
```

The Raspberry Pi capture and upload stages are part of the intended system, but their source and deployment configuration were not found in this workspace. Their capture interval, retry behavior, and camera dependencies cannot yet be documented or verified.

| Location | Purpose |
| --- | --- |
| `AI Model/src/cloud_detect.py` | Continuous Cloudinary detection and CSV logging |
| `AI Model/src/detect.py` | Local JPEG inference and console output |
| `AI Model/src/cloud_detect_v2.py` | Empty placeholder; no runnable implementation |
| `AI Model/models/best.pt` | Locally supplied model weights; excluded from Git |
| `AI Model/images/` | Local datasets, input images, annotated output, and processing state; excluded |
| `AI Model/logs/` | Monthly detection CSV files; excluded |
| `Project/lesvos_fire_camera_optimizer/` | QGIS camera-placement algorithm variants and existing instructions |
| `Project/` | Also contains local QGIS projects and spatial data; excluded except optimizer source/docs |
| `Data/`, `Reports/`, `Results/` | Local supporting material; excluded |

### Cloud processing

1. Load credentials from `AI Model/.env` and weights from `AI Model/models/best.pt`.
2. Search Cloudinary for the newest 20 images matching `asset_folder:Lesvos_Fire_Detection/* AND resource_type:image`.
3. Process that batch oldest first, skipping asset IDs in `AI Model/images/cloud_results/processed.json`.
4. Download each original and run CPU inference with confidence `0.25` and image size `960`. Class labels come from the supplied model.
5. Save an annotated JPEG when detections exist, update processing state, append CSV rows, and delete the downloaded original after successful logging.
6. Catch loop errors, print them, and wait 60 seconds before polling again. Stop with Ctrl+C.

CSV files use the asset creation date: `AI Model/logs/YYYY/detections_YYYY-MM.csv`. Each detection gets one row; images without detections get a `NONE` row. Camera ID is the final folder component, falling back to `UNKNOWN` when folder metadata is absent.

```text
camera_id,timestamp,asset_id,detection,confidence,x1,y1,x2,y2,image_url
```

Coordinates are model bounding-box pixel coordinates. Confidence is rounded to four decimals and coordinates to one decimal. CSVs include source image URLs and should remain private.

## Setup

Use a Python environment compatible with the supplied model and dependencies. The requirements reflect versions found in the existing local environment; a fresh installation has not been validated. Model provenance, training data, class mapping, accuracy, and redistribution rights are not documented in the available source.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example 'AI Model/.env'
```

On Linux, activate with `source .venv/bin/activate` and copy with `cp .env.example 'AI Model/.env'`.

Fill in the three values in `AI Model/.env` locally. Use credentials for a Cloudinary account permitted to search the relevant assets. The root `.env.example` is only a template: the current script explicitly loads `AI Model/.env`.

Place your trusted, compatible weights at `AI Model/models/best.pt`. Weights are deliberately not distributed or downloaded by this repository. Upload camera images into Cloudinary asset folders such as `Lesvos_Fire_Detection/CAMERA_01`; capture/upload code must be supplied separately.

Run cloud processing from the repository root:

```powershell
python 'AI Model/src/cloud_detect.py'
```

For local inference, create `AI Model/images/test` and `AI Model/images/results`, put test `.jpg` files in the former, then run from `AI Model` because this script uses working-directory-relative paths:

```powershell
Set-Location 'AI Model'
python src/detect.py
```

`detect.py` imports `huggingface_hub` but does not download weights. QGIS optimization scripts run within an appropriately configured QGIS environment with NumPy, GDAL, and Processing; they are separate from the detection requirements. See [existing optimizer instructions](Project/lesvos_fire_camera_optimizer/README_GR.md). Those instructions may refer to older filenames; the workspace contains multiple algorithm variants, with no designated canonical version.

## Known limitations from code review

- Cloud search has no pagination: images outside the newest 20 can be missed during backlogs or downtime.
- Processing state is saved before CSV logging. A logging failure can leave an asset marked processed without a CSV record. CSV and state writes are not transactional.
- State and downloads have no retention policy; use one worker to avoid concurrent writes.
- The console output directly accesses `asset_folder`, although CSV camera extraction has a fallback. Missing metadata can interrupt an iteration.
- `detect.py` repeats the last image's detections after its loop and raises an error when no input images exist.
- There is no implemented alert delivery or verified end-to-end camera deployment in the available source. Detection performance has not been evaluated in this review.

These observations are documented for follow-up; existing Python behavior has not been changed.

## Proposed future repository layout

The following is a proposal only. No files have been moved or renamed.

```text
lesvos-fire-detection/
  README.md
  .gitignore
  .env.example
  requirements.txt
  src/
    detection/
      cloud_detect.py
      detect.py
  raspberry_pi/              # add verified capture/upload source when available
  tools/
    camera_placement/        # selected QGIS scripts and instructions
  docs/                     # deployment and model provenance documentation
  models/                   # local only
  images/                   # local only
  logs/                     # local only
```

Moving detection scripts changes how their base directories resolve. A later reorganization must update path handling, environment loading, and run instructions together and verify behavior. Choose a maintained QGIS variant before consolidating its history. For now, the existing layout remains usable.

## Repository hygiene and review

The root ignore file uses an allowlist: only these four setup/documentation files, detection Python source, and optimizer Python/Markdown/text documentation are candidates for publication. Other local material is excluded by default. It also excludes environment files, virtual environments, images, datasets, CSVs, logs, weights, and credential-like filenames.

Ignore rules do not remove already tracked files or secrets in Git history. Before any first commit or push, inspect the exact staged file list and staged diff, scan the candidate source for credentials, and inspect any existing history. Rotate any credential previously published. Never paste credentials or private image URLs into issues or documentation.

Review the proposed layout and the exact staged files before approving publication. No license has been selected; establish ownership and dependency/model licensing before distributing project assets.
