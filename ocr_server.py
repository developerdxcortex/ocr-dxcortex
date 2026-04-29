"""
FastAPI OCR server for ultrasound images.

Wraps extract_ultrasound.py with two endpoints:
  POST /ocr        — process an image (DICOM via Orthanc, OR base64 image)
  GET  /health     — readiness check

Response shape matches what SRDataPanel.tsx expects:
{
  "organ":        "LEFT KIDNEY" | null,
  "measurements": [{"label": "D", "value": "10.79cm", "index": 1}, ...],
  "count":        2,
  "raw_text":     "...",
  "resolution":   "1136x852",
  "source":       "dicom" | "image",
  "error":        null
}

Run locally:
    pip install -r requirements.txt
    uvicorn ocr_server:app --host 0.0.0.0 --port 8000 --reload

Then in your Vite dev server, proxy /api/ocr-proxy → http://localhost:8000/ocr
(see the README block at the bottom of this file).
"""

import base64
import io
import logging
import os
import time
from typing import Any, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# Reuse all the parsing logic from your existing script — sirf import karna hai
from extract_ultrasound import extract_measurements, extract_organ_label

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ocr_server")


# ─── FastAPI app + CORS ──────────────────────────────────────────────────────
app = FastAPI(title="Ultrasound OCR Server", version="1.0.0")

# In dev, allow everything.  Tighten this for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response models ─────────────────────────────────────────────────
class OCRRequest(BaseModel):
    """Either DICOM mode (instance_id + orthanc_url) or image mode (base64)."""

    # DICOM mode
    instance_id: Optional[str] = None
    orthanc_url: Optional[str] = None
    orthanc_username: Optional[str] = None
    orthanc_password: Optional[str] = None

    # Image fallback mode
    image: Optional[str] = None  # base64-encoded image bytes (no data URI prefix)


class Measurement(BaseModel):
    label: str
    value: str  # combined "{value}{unit}" like "10.79cm" — what the frontend expects
    index: Optional[int] = None


class OCRResponse(BaseModel):
    organ: Optional[str] = None
    measurements: list[Measurement] = []
    count: int = 0
    raw_text: str = ""
    resolution: str = ""
    source: str = ""  # "dicom" | "image"
    error: Optional[str] = None


# ─── Orthanc fetch ───────────────────────────────────────────────────────────
def fetch_dicom_image_from_orthanc(
    orthanc_url: str,
    instance_id: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Image.Image:
    """Fetch the rendered preview of a DICOM instance from Orthanc.

    Tries /rendered first (JPEG, full resolution) and falls back to /preview
    (PNG, smaller).  Both return a PIL Image ready for OCR.
    """
    base = orthanc_url.rstrip("/")
    auth = (username, password) if username else None

    last_err: Optional[Exception] = None
    for endpoint in (f"/instances/{instance_id}/rendered", f"/instances/{instance_id}/preview"):
        url = f"{base}{endpoint}"
        try:
            log.info("Fetching DICOM: %s", url)
            r = requests.get(url, auth=auth, timeout=30)
            if r.status_code != 200:
                log.warning("  -> HTTP %d", r.status_code)
                continue
            if len(r.content) < 500:
                log.warning("  -> response too small (%d bytes), skipping", len(r.content))
                continue
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            log.info("  -> success: %dx%d", img.width, img.height)
            return img
        except Exception as e:
            last_err = e
            log.warning("  -> error: %s", e)
            continue

    raise HTTPException(
        status_code=502,
        detail=f"Could not fetch DICOM {instance_id} from Orthanc: {last_err}",
    )


def decode_base64_image(b64: str) -> Image.Image:
    """Decode a base64 string (with or without data URI prefix) to a PIL Image."""
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")


# ─── Core processing ─────────────────────────────────────────────────────────
def process_image(img: Image.Image, source: str) -> OCRResponse:
    """Run the extraction pipeline on a PIL Image and shape the result for the frontend."""
    t0 = time.time()
    measurements_raw = extract_measurements(img)
    organ = extract_organ_label(img)
    if organ is None:
        # Fallback: derive from measurement labels (e.g. "FIB" → "TIB/FIB")
        from extract_ultrasound import derive_organ_from_measurements
        organ = derive_organ_from_measurements(measurements_raw)

    # Combine value + unit into one display string ("10.79" + "cm" -> "10.79cm")
    measurements: list[Measurement] = []
    for m in measurements_raw:
        value_str = f"{m['value']}{m.get('unit', '')}".strip()
        measurements.append(
            Measurement(
                label=m["label"],
                value=value_str,
                index=m.get("index"),
            )
        )

    elapsed_ms = int((time.time() - t0) * 1000)
    log.info(
        "Processed [%s] %dx%d in %dms — organ=%s, %d measurements",
        source, img.width, img.height, elapsed_ms, organ, len(measurements),
    )

    return OCRResponse(
        organ=organ,
        measurements=measurements,
        count=len(measurements),
        raw_text="",  # extract_ultrasound doesn't surface this, leaving empty
        resolution=f"{img.width}x{img.height}",
        source=source,
    )


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "ultrasound-ocr"}


@app.post("/ocr", response_model=OCRResponse)
def ocr_endpoint(req: OCRRequest) -> OCRResponse:
    """Run OCR on a DICOM instance (preferred) or a base64 image (fallback)."""
    # ── Mode 1: DICOM via Orthanc ────────────────────────────────────────────
    if req.instance_id and req.orthanc_url:
        try:
            img = fetch_dicom_image_from_orthanc(
                orthanc_url=req.orthanc_url,
                instance_id=req.instance_id,
                username=req.orthanc_username,
                password=req.orthanc_password,
            )
            return process_image(img, source="dicom")
        except HTTPException:
            # If DICOM fetch failed but we have a base64 fallback, try that
            if req.image:
                log.info("DICOM fetch failed, falling back to base64 image")
            else:
                raise

    # ── Mode 2: base64 image ─────────────────────────────────────────────────
    if req.image:
        img = decode_base64_image(req.image)
        return process_image(img, source="image")

    raise HTTPException(
        status_code=400,
        detail="Provide either (instance_id + orthanc_url) or image (base64).",
    )


# ─── README ──────────────────────────────────────────────────────────────────
"""
=== HOW TO RUN LOCALLY ===

1. Put this file (ocr_server.py) and your existing extract_ultrasound.py in the
   same folder.

2. Install dependencies:
       pip install -r requirements.txt

3. Start the server:
       uvicorn ocr_server:app --host 0.0.0.0 --port 8000 --reload

4. Configure Vite (your frontend dev server) to proxy /api/ocr-proxy to the
   FastAPI server.  In vite.config.ts:

       export default defineConfig({
         // ...
         server: {
           proxy: {
             '/api/ocr-proxy': {
               target: 'http://localhost:8000',
               changeOrigin: true,
               rewrite: (path) => path.replace(/^\\/api\\/ocr-proxy/, '/ocr'),
             },
             // keep your existing /api/orthanc-proxy here too
           },
         },
       });

5. Test:
       curl http://localhost:8000/health
       curl -X POST http://localhost:8000/ocr \\
         -H 'Content-Type: application/json' \\
         -d '{"instance_id": "abc-123", "orthanc_url": "https://thethetanap.com",
              "orthanc_username": "user", "orthanc_password": "pass"}'

=== DEPLOYING TO EC2 ===

When ready, on EC2:
    sudo apt install -y python3-pip tesseract-ocr
    pip install -r requirements.txt
    # Run with gunicorn behind nginx, or use systemd:
    uvicorn ocr_server:app --host 0.0.0.0 --port 8000 --workers 2

Then update VITE_API_BASE_URL or your reverse proxy to point /api/ocr-proxy
at the EC2 instance.
"""