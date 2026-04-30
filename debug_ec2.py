"""
Debug script: dump RAW Tesseract OCR output for problematic patient images.

USAGE:
  1. Frontend Network tab kholo, problematic image ka instance_id copy karo
     (POST /api/ocr-proxy ka payload mein dikhta hai)
  2. Edit INSTANCE_IDS list below, add 1-5 instance IDs
  3. Run on EC2:
       cd ~/ocr
       source venv/bin/activate
       python3 debug_ec2.py
  4. Output Claude ko bhejo — fir wo exact OCR pattern dekh ke fix de sakta hai
"""
import sys
import io
import urllib3

import requests
from PIL import Image
import pytesseract

urllib3.disable_warnings()
sys.path.insert(0, "/home/ubuntu/ocr")
import extract_ultrasound as eu  # noqa: E402

ORTHANC_URL = "https://thetanap.com"
USERNAME = "orthanc"
PASSWORD = "orthanc"

# ──────────────────────────────────────────────────────────────────────
# EDIT THIS LIST: paste instance IDs from frontend Network tab payloads
# ──────────────────────────────────────────────────────────────────────
INSTANCE_IDS = [
    # "1234abcd-5678efgh-...",
    # "another-instance-id-here",
]


def dump_image(instance_id: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"INSTANCE: {instance_id}")
    print("=" * 70)

    url = f"{ORTHANC_URL}/instances/{instance_id}/rendered"
    r = requests.get(url, auth=(USERNAME, PASSWORD), timeout=30, verify=False)
    if r.status_code != 200:
        print(f"  Failed: HTTP {r.status_code}")
        return
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    print(f"  Image: {img.size}")
    w, h = img.size

    # Yellow organ label OCR
    print("\n  --- Yellow organ label ---")
    scan_area = img.crop(
        (int(w * 0.15), int(h * 0.10), int(w * 0.90), int(h * 0.90))
    )
    for thresh in (eu.STRICT_YELLOW, eu.LOOSE_YELLOW):
        yimg = eu.upscale(eu.isolate_color(scan_area, thresh), 3)
        text = pytesseract.image_to_string(yimg, config="--psm 6").strip()
        print(f"    Threshold {thresh}: {text!r}")

    # Standard measurement crops
    print("\n  --- Standard measurement crops ---")
    for left, top, right, bottom in eu.NORMAL_CROPS:
        crop = img.crop(
            (int(w * left), int(h * top), int(w * right), int(h * bottom))
        )
        text_n = eu.ocr_measurement_crop(crop, inverted=False)
        text_i = eu.ocr_measurement_crop(crop, inverted=True)
        print(f"    Crop ({left}, {top}, {right}, {bottom})  size={crop.size}")
        print(f"      Normal:   {text_n!r}")
        print(f"      Inverted: {text_i!r}")

    # Extra debug crops to see if measurement box is outside standard area
    print("\n  --- Extra debug crops ---")
    extra_crops = [
        ("Very tight bottom", (0.85, 0.93, 1.00, 1.00)),
        ("Tighter top",       (0.85, 0.88, 1.00, 0.96)),
        ("Wider bottom",      (0.65, 0.85, 1.00, 1.00)),
        ("Wide+tall",         (0.60, 0.55, 1.00, 1.00)),
        ("Full bottom row",   (0.00, 0.85, 1.00, 1.00)),
    ]
    for label, (left, top, right, bottom) in extra_crops:
        crop = img.crop(
            (int(w * left), int(h * top), int(w * right), int(h * bottom))
        )
        text_n = eu.ocr_measurement_crop(crop, inverted=False)
        text_i = eu.ocr_measurement_crop(crop, inverted=True)
        print(f"    {label} ({left}, {top}, {right}, {bottom})  size={crop.size}")
        print(f"      Normal:   {text_n!r}")
        print(f"      Inverted: {text_i!r}")

    # Final extraction result
    print("\n  --- Current pipeline result ---")
    organ = eu.extract_organ_label(img)
    measurements = eu.extract_measurements(img)
    print(f"    Organ: {organ}")
    print(f"    Measurements ({len(measurements)}):")
    for m in measurements:
        print(f"      {m}")


def main() -> None:
    if not INSTANCE_IDS:
        print("Edit script and add INSTANCE_IDS from frontend Network tab.")
        print("Look for /api/ocr-proxy POST requests and copy 'instance_id' values.")
        sys.exit(1)
    for iid in INSTANCE_IDS:
        dump_image(iid)


if __name__ == "__main__":
    main()