import os
import subprocess
import time
from pathlib import Path

# Cloudinary settings
CLOUD_NAME = os.environ["CLOUDINARY_CLOUD_NAME"]
UPLOAD_PRESET = os.environ["CLOUDINARY_UPLOAD_PRESET"]
CLOUD_FOLDER = os.environ.get("CLOUDINARY_ASSET_FOLDER", "Lesvos_Fire_Detection/Camera_001")
DELAY_SECONDS = 60

while True:
    photo = Path(f"/tmp/fire_{int(time.time())}.jpg")

    # Capture with manual focus at infinity
    subprocess.run([
        "rpicam-still", "-n", "-t", "2000",
        "--autofocus-mode", "manual",
        "--lens-position", "0",
        "-o", str(photo)
    ], check=True)

    # Upload only via Wi-Fi (wlan0)
    subprocess.run([
        "curl", "-4", "--interface", "if!wlan0",
        "--fail-with-body", "-sS",
        "-X", "POST",
        f"https://api.cloudinary.com/v1_1/{CLOUD_NAME}/image/upload",
        "-F", f"file=@{photo}",
        "-F", f"upload_preset={UPLOAD_PRESET}",
        "-F", f"asset_folder={CLOUD_FOLDER}"
    ], check=True)

    # Delete the local photo after successful upload
    photo.unlink()
    print("Photo uploaded successfully!")
    time.sleep(DELAY_SECONDS)
