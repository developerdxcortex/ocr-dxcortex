"""
Test multiple Tesseract PSM modes on the FEMUR image to see if any of them
read "75" correctly instead of "15".

PSM modes:
  3  = Fully automatic page segmentation (default)
  6  = Assume uniform block of text (current)
  7  = Treat as single text line
  8  = Treat as single word
  11 = Sparse text
  13 = Raw line — bypass tesseract heuristics
"""
import sys, requests, io, urllib3
urllib3.disable_warnings()
sys.path.insert(0, '/home/ubuntu/ocr')
from PIL import Image, ImageOps
import pytesseract
import extract_ultrasound as eu

INSTANCE_ID = "c13262c0-60254868-bb3a1288-ad94d16f-3c70478b"
ORTHANC_URL = "https://thetanap.com"

url = f"{ORTHANC_URL}/instances/{INSTANCE_ID}/rendered"
r = requests.get(url, auth=("orthanc", "orthanc"), timeout=30, verify=False)
img = Image.open(io.BytesIO(r.content)).convert("RGB")
w, h = img.size
print(f"Image: {img.size}\n")

# Use the standard "tight" crop where measurement box is
crop = img.crop((int(w*0.81), int(h*0.86), int(w*1.0), int(h*1.0)))

# Try at different upscale factors and PSM modes
for upscale_factor in [3, 4, 5, 6]:
    upscaled = crop.resize((crop.width * upscale_factor, crop.height * upscale_factor), Image.LANCZOS)
    print(f"\n=== Upscale {upscale_factor}x (size={upscaled.size}) ===")
    for psm in [6, 7, 8, 11, 12, 13]:
        try:
            text = pytesseract.image_to_string(upscaled, config=f"--psm {psm}")
            text_clean = text.strip().replace('\n', ' | ')
            print(f"  PSM {psm:2d}: {text_clean!r}")
        except Exception as e:
            print(f"  PSM {psm:2d}: ERROR {e}")

# Also try inverted
print("\n\n========== INVERTED ==========")
crop_inv = ImageOps.invert(crop.convert("L"))
for upscale_factor in [3, 4, 5]:
    upscaled = crop_inv.resize((crop.width * upscale_factor, crop.height * upscale_factor), Image.LANCZOS)
    print(f"\n=== Inverted Upscale {upscale_factor}x ===")
    for psm in [6, 7, 8, 11, 13]:
        try:
            text = pytesseract.image_to_string(upscaled, config=f"--psm {psm}")
            text_clean = text.strip().replace('\n', ' | ')
            print(f"  PSM {psm:2d}: {text_clean!r}")
        except Exception as e:
            print(f"  PSM {psm:2d}: ERROR {e}")

# Try with character whitelist
print("\n\n========== WITH CHARACTER WHITELIST ==========")
upscaled = crop.resize((crop.width * 4, crop.height * 4), Image.LANCZOS)
for psm in [6, 7, 11]:
    config = f"--psm {psm} -c tessedit_char_whitelist=0123456789.cmFLBPDHCAGwd%/ "
    text = pytesseract.image_to_string(upscaled, config=config)
    text_clean = text.strip().replace('\n', ' | ')
    print(f"  PSM {psm} + whitelist: {text_clean!r}")
