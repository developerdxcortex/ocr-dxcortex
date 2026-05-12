"""
Diagnose why FEMUR FL/BPD still returns 15% even after binarized pass.
Run on EC2 to see what each pass produces.
"""
import sys, requests, io, urllib3
urllib3.disable_warnings()
sys.path.insert(0, '/home/ubuntu/ocr')
from PIL import Image
import extract_ultrasound as eu

INSTANCE_ID = "c13262c0-60254868-bb3a1288-ad94d16f-3c70478b"
url = f"https://thetanap.com/instances/{INSTANCE_ID}/rendered"
r = requests.get(url, auth=("orthanc", "orthanc"), timeout=30, verify=False)
img = Image.open(io.BytesIO(r.content)).convert("RGB")
print(f"Image: {img.size}\n")

# Manually run each pass and show results
print("=" * 60)
print("PASS 1 — NORMAL on standard crops")
print("=" * 60)
normal_results = eu._extract_measurements_pass(img, eu.NORMAL_CROPS, mode="normal")
for m in normal_results:
    print(f"  {m}")

print("\n" + "=" * 60)
print("PASS 3 — BINARIZED on standard crops")
print("=" * 60)
bin_results = eu._extract_measurements_pass(img, eu.NORMAL_CROPS, mode="binarized")
for m in bin_results:
    print(f"  {m}")

# Show raw OCR text for binarized mode (what Tesseract returned with binarization)
print("\n" + "=" * 60)
print("RAW BINARIZED OCR TEXT (per crop)")
print("=" * 60)
w, h = img.size
for left, top, right, bottom in eu.NORMAL_CROPS:
    crop = img.crop((int(w*left), int(h*top), int(w*right), int(h*bottom)))
    text = eu.ocr_measurement_crop(crop, mode="binarized")
    print(f"\nCrop ({left}, {top}, {right}, {bottom}):")
    print(f"  Raw: {text!r}")

# Now run full pipeline
print("\n" + "=" * 60)
print("FULL PIPELINE (extract_measurements)")
print("=" * 60)
final = eu.extract_measurements(img)
for m in final:
    print(f"  {m}")
