"""
Test image preprocessing techniques to improve digit recognition.
Try: sharpening, contrast boost, binarization, larger upscale.
"""
import sys, requests, io, urllib3
urllib3.disable_warnings()
sys.path.insert(0, '/home/ubuntu/ocr')
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract

INSTANCE_ID = "c13262c0-60254868-bb3a1288-ad94d16f-3c70478b"
url = f"https://thetanap.com/instances/{INSTANCE_ID}/rendered"
r = requests.get(url, auth=("orthanc", "orthanc"), timeout=30, verify=False)
img = Image.open(io.BytesIO(r.content)).convert("RGB")
w, h = img.size
crop = img.crop((int(w*0.81), int(h*0.86), int(w*1.0), int(h*1.0)))
print(f"Crop size: {crop.size}\n")

def show(label, processed_img):
    text = pytesseract.image_to_string(processed_img, config="--psm 6").strip().replace('\n', ' | ')
    has_75 = '75' in text and '15' not in text.replace('15%', '')
    flag = '★ HAS 75!' if has_75 else ('15' in text and 'has 15' or '')
    print(f"  {label:50} -> {text!r}  {flag}")

# Baseline
print("=== Baseline ===")
show("3x LANCZOS", crop.resize((crop.width*3, crop.height*3), Image.LANCZOS))

print("\n=== Greyscale + binarize ===")
gray = crop.convert("L")
for thresh in [80, 100, 120, 140, 160, 180]:
    binarized = gray.point(lambda p: 255 if p > thresh else 0)
    upscaled = binarized.resize((binarized.width*4, binarized.height*4), Image.LANCZOS)
    show(f"binarize thresh={thresh}, 4x", upscaled)

print("\n=== Inverted binarize (white-on-dark text) ===")
gray = crop.convert("L")
inverted = ImageOps.invert(gray)
for thresh in [80, 100, 120, 140, 160, 180]:
    binarized = inverted.point(lambda p: 255 if p > thresh else 0)
    upscaled = binarized.resize((binarized.width*4, binarized.height*4), Image.LANCZOS)
    show(f"INV binarize thresh={thresh}, 4x", upscaled)

print("\n=== Sharpening ===")
gray = crop.convert("L")
upscaled = gray.resize((gray.width*4, gray.height*4), Image.LANCZOS)
for radius in [0.5, 1.0, 2.0]:
    for amount in [50, 100, 200]:
        sharpened = upscaled.filter(ImageFilter.UnsharpMask(radius=radius, percent=amount))
        show(f"sharpen r={radius} a={amount}", sharpened)

print("\n=== Contrast boost ===")
gray = crop.convert("L")
upscaled = gray.resize((gray.width*4, gray.height*4), Image.LANCZOS)
for factor in [1.5, 2.0, 3.0]:
    enhanced = ImageEnhance.Contrast(upscaled).enhance(factor)
    show(f"contrast x{factor}", enhanced)

print("\n=== Combined: invert + binarize + sharpen ===")
gray = crop.convert("L")
inverted = ImageOps.invert(gray)
for thresh in [100, 120, 140]:
    binarized = inverted.point(lambda p: 255 if p > thresh else 0)
    upscaled = binarized.resize((binarized.width*5, binarized.height*5), Image.LANCZOS)
    sharpened = upscaled.filter(ImageFilter.SHARPEN)
    show(f"INV + bin t={thresh} + 5x + sharpen", sharpened)

print("\n=== With dilate/erode (morphology approximation) ===")
gray = crop.convert("L")
inverted = ImageOps.invert(gray)
for thresh in [100, 130]:
    binarized = inverted.point(lambda p: 255 if p > thresh else 0)
    # Apply MaxFilter (dilate) to thicken thin strokes
    for size in [3, 5]:
        thickened = binarized.filter(ImageFilter.MaxFilter(size))
        upscaled = thickened.resize((thickened.width*4, thickened.height*4), Image.LANCZOS)
        show(f"INV t={thresh} dilate{size} 4x", upscaled)
