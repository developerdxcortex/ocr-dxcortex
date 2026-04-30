"""
Extract organ label + measurements from ultrasound images. (v12 — final)

Supports:
  - Standard organ scans (kidney, liver, prostate, spleen, pancreas, bladder, etc.)
  - Fetal biometry (HUMERUS, FEMUR, BPD, HC, AC, FL, ORBITS, TIB/FIB, etc.)
  - Doppler studies (Rt Ut-PS, Rt Ut-PI, MCA, UA, etc.)
  - M-mode (HR/heart rate)

Strategies:
  - Yellow-pixel isolation + 3x upscaling for the organ label
  - Three-pass organ matching: keyword substring → fuzzy keyword → fuzzy whole-name
  - Multiple measurement crops (tight/medium/wide) for different box sizes
  - Both normal and inverted-color OCR run in parallel; inversion catches
    small white-on-dark text where "." is misread as "A" or letter
  - Decimal recovery for length values (445→4.45) and ratios (132→1.32)
  - Strict junk filters reject OCR garbage (lowercase-uppercase transitions,
    non-clinical characters, partial-number labels)

Usage:
    python extract_ultrasound.py path/to/image.jpg
"""

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageOps

# Windows: uncomment if tesseract.exe is not on your PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ─── Known organ / annotation labels (for the yellow text on scan) ─────────
KNOWN_ORGANS = [
    # Abdomen
    "RIGHT KIDNEY", "LEFT KIDNEY",
    "LIVER", "SPLEEN", "PANCREAS",
    "GALL BLADDER", "GALLBLADDER",
    "URINARY BLADDER", "BLADDER",
    "PROSTATE", "UTERUS", "OVARY",
    "RIGHT OVARY", "LEFT OVARY",
    "AORTA", "IVC", "CBD", "PORTAL VEIN",
    "THYROID", "RIGHT LOBE", "LEFT LOBE",
    # Fetal biometry / parts
    "HUMERUS", "FEMUR", "RADIUS", "ULNA",
    "TIBIA", "FIBULA", "TIB/FIB", "TIB FIB",
    "SPINE", "FACE", "ORBITS", "EYES",
    "HEART", "STOMACH", "PLACENTA",
    "BPD", "HC", "AC", "FL", "NT", "NB", "NUCHAL FOLD",
    # Doppler vessels
    "UT A", "RT UT A", "LT UT A",
    "MCA", "UA", "DV", "UMB A", "UMB CORD",
]

# Short codes (≤4 chars) — matched as substrings rather than fuzzy
SHORT_ORGAN_CODES = ["UT A", "MCA", "UA", "DV"]

ORGAN_FUZZY_THRESHOLD = 0.7

# Distinctive keywords that strongly indicate a particular organ — even when
# OCR mangles surrounding text.  If a candidate string CONTAINS one of these
# as a substring, we boost the corresponding organ match heavily.  This is
# what separates real "ITF 1 KIDNEY" from noise like "ATA".
ORGAN_KEYWORDS = {
    "KIDNEY": ["RIGHT KIDNEY", "LEFT KIDNEY"],
    "LIVER": ["LIVER"],
    "SPLEEN": ["SPLEEN"],
    "PANCREAS": ["PANCREAS"],
    "PANCRE": ["PANCREAS"],   # OCR often clips the trailing "AS"
    "BLADDER": ["BLADDER"],   # plain BLADDER — the image often just says "BLADDER"
    "GALL": ["GALL BLADDER"],
    "URINARY": ["URINARY BLADDER"],
    "PROSTATE": ["PROSTATE"],
    "UTERUS": ["UTERUS"],
    "OVARY": ["RIGHT OVARY", "LEFT OVARY", "OVARY"],
    "AORTA": ["AORTA"],
    "THYROID": ["THYROID"],
    "HUMERUS": ["HUMERUS"],
    "FEMUR": ["FEMUR"],
    "TIBIA": ["TIBIA"],
    "FIBULA": ["FIBULA"],
    "ORBITS": ["ORBITS"],
    "PLACENTA": ["PLACENTA"],
    "STOMACH": ["STOMACH"],
}

# ─── Known Doppler measurement codes (used to fuzzy-correct OCR misreads) ──
DOPPLER_CODES = [
    "Rt Ut-PS", "Rt Ut-ED", "Rt Ut-S/D", "Rt Ut-PI", "Rt Ut-RI",
    "Rt Ut-MD", "Rt Ut-TAmax", "Rt Ut-HR",
    "Lt Ut-PS", "Lt Ut-ED", "Lt Ut-S/D", "Lt Ut-PI", "Lt Ut-RI",
    "Lt Ut-MD", "Lt Ut-TAmax", "Lt Ut-HR",
    "MCA-PS", "MCA-ED", "MCA-S/D", "MCA-PI", "MCA-RI", "MCA-HR",
    "UA-PS", "UA-ED", "UA-S/D", "UA-PI", "UA-RI", "UA-HR",
]
DOPPLER_CORRECT_THRESHOLD = 0.7

# Yellow thresholds for annotation extraction
STRICT_YELLOW = (200, 200, 80)
LOOSE_YELLOW  = (180, 180, 120)

# Crops (left, top, right, bottom as fractions of W/H).  Listed in order of
# trust — tighter, narrower crops produce cleaner OCR.  When the SAME label
# appears in a later (less-trusted) crop with a DIFFERENT value, the later
# reading is discarded as a likely OCR error.
# Normal crops — tried first, in order, with regular OCR
NORMAL_CROPS = [
    (0.81, 0.86, 1.00, 1.00),   # tight — small boxes (kidney, liver, prostate)
    (0.83, 0.78, 1.00, 1.00),   # medium — tall fetal biometry boxes (AC/EFW/HC etc.)
    (0.78, 0.65, 1.00, 1.00),   # wide — Doppler panels with long labels
]
# Fallback crops — only used when normal OCR finds NOTHING.
# Kept short to avoid runaway latency on images that simply have no
# measurement panel (e.g. fetal scan with only the yellow organ label).
#   - First crop: very-tight + inverted (handles white-on-dark single-line
#     boxes: rukaina3 SPLEEN D 6.73cm, rukaina5 LIVER D 11.33cm).
#   - Second crop: wider — catches measurement boxes that sit slightly
#     left of the tight 0.81 boundary (some machine layouts).
INVERTED_FALLBACK_CROPS = [
    (0.85, 0.93, 1.00, 1.00),
    (0.70, 0.85, 1.00, 1.00),
]

UNITS_LIST = ["cm/s", "mm/s", "cm", "mm", "bpm", "kg", "ml", "%", "g", "s"]
UNITS = "(?:" + "|".join(re.escape(u) for u in UNITS_LIST) + ")"


# ─── Image helpers ──────────────────────────────────────────────────────────
def upscale(img, factor=3):
    return img.resize((img.width * factor, img.height * factor), Image.LANCZOS)


def isolate_color(img, threshold):
    r_min, g_min, b_max = threshold
    arr = np.array(img.convert("RGB"))
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = (r > r_min) & (g > g_min) & (b < b_max)
    out = np.full_like(arr, 255)
    out[mask] = [0, 0, 0]
    return Image.fromarray(out.astype(np.uint8))


# ─── Organ matching ─────────────────────────────────────────────────────────
def find_short_code(text):
    upper = re.sub(r"[^A-Z\s]", " ", text.upper())
    upper = re.sub(r"\s+", " ", upper)
    for code in SHORT_ORGAN_CODES:
        if re.search(rf"\b{re.escape(code)}\b", upper):
            return code
    return None


def fuzzy_match_organ(text):
    if not text:
        return None, 0.0
    # Clean per-line so a short code like "UT A" stays separable from waveform noise
    lines = []
    for line in text.upper().splitlines():
        cleaned = re.sub(r"[^A-Z\s/]", " ", line)
        cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
        if cleaned:
            lines.append(cleaned)
    if not lines:
        return None, 0.0

    candidates = lines + [" ".join(lines)]

    # FIRST PASS: high-confidence keyword substring match.  If a distinctive
    # organ keyword (KIDNEY, LIVER, PROSTATE, ...) appears as a substring of
    # any candidate, prefer that match.  This catches OCR results like
    # "ITF 1 KIDNEY" reliably while ignoring pure noise like "ATA".
    for cand in candidates:
        for keyword, organ_options in ORGAN_KEYWORDS.items():
            if keyword in cand:
                if len(organ_options) == 1:
                    return organ_options[0], 1.0
                # Disambiguate (e.g. "RIGHT KIDNEY" vs "LEFT KIDNEY").
                # Direction-keyword preference: if the candidate contains
                # "RIGHT" anywhere (including word-reversed forms like
                # "KIDNEY RIGHT"), pick the RIGHT option. Same for LEFT.
                # SequenceMatcher alone is unreliable for word-reversed
                # text — "KIDNEY RIGHT" actually scores higher against
                # "LEFT KIDNEY" than "RIGHT KIDNEY" because of how it
                # measures matching subsequences.
                has_right = "RIGHT" in cand or " RT " in (" " + cand + " ") or cand.startswith("RT ")
                has_left  = "LEFT"  in cand or " LT " in (" " + cand + " ") or cand.startswith("LT ")
                if has_right and not has_left:
                    for opt in organ_options:
                        if opt.startswith("RIGHT") or opt.startswith("RT "):
                            return opt, 1.0
                if has_left and not has_right:
                    for opt in organ_options:
                        if opt.startswith("LEFT") or opt.startswith("LT "):
                            return opt, 1.0
                # No clear direction keyword — fall back to fuzzy ranking.
                best = max(
                    organ_options,
                    key=lambda o: SequenceMatcher(None, o, cand).ratio()
                )
                return best, 1.0

    # SECOND PASS: fuzzy keyword recovery.  OCR sometimes mangles letters
    # ("BLADDER" → "B: ADIER", "SPLEEN" → "SELEEN").  We try two strategies:
    #   (a) collapse all non-letters in the candidate and fuzzy-match against
    #       the keyword with strict length tolerance (|diff| ≤ 1)
    #   (b) suffix match: any 4+ letter word whose last 3 chars match the
    #       keyword's last 3 chars (catches "RULANEY" → "KIDNEY" via "NEY")
    # We pick the keyword with the HIGHEST score across all candidates so
    # that "HUME RUS" → HUMERUS (score 1.0) wins over UTERUS (score 0.77).
    fuzzy_best_score = 0.0
    fuzzy_best_organ = None
    fuzzy_best_cand = None
    for cand in candidates:
        cand_compact = re.sub(r"[^A-Z]", "", cand)
        cand_words = [w for w in cand.split() if len(w) >= 4]
        for keyword, organ_options in ORGAN_KEYWORDS.items():
            score = 0.0
            # Strategy (a): compact fuzzy
            if abs(len(cand_compact) - len(keyword)) <= 1:
                s = SequenceMatcher(None, cand_compact, keyword).ratio()
                if s >= 0.7:
                    score = s
            # Strategy (b): 3-char suffix — assign moderate score so it can
            # match when compact fuzzy fails but doesn't beat a real match
            if score == 0.0:
                kw_suffix = keyword[-3:]
                for word in cand_words:
                    if word[-3:] == kw_suffix:
                        score = 0.8
                        break
            if score > fuzzy_best_score:
                fuzzy_best_score = score
                fuzzy_best_organ = organ_options
                fuzzy_best_cand = cand
    if fuzzy_best_organ is not None:
        if len(fuzzy_best_organ) == 1:
            return fuzzy_best_organ[0], fuzzy_best_score
        best = max(
            fuzzy_best_organ,
            key=lambda o: SequenceMatcher(None, o, fuzzy_best_cand).ratio()
        )
        return best, fuzzy_best_score

    # THIRD PASS: regular fuzzy matching for short codes / edge cases
    best_score, best_match = 0.0, None
    for organ in KNOWN_ORGANS:
        is_short = len(organ) <= 4
        # Long organs require very high similarity now (keyword pass handles
        # the easy cases) so noise like "ATA" can't slip through to AORTA.
        threshold_for_organ = 0.75 if is_short else 0.85
        # Length filter — count NON-WHITESPACE chars in candidate so "I CU F" (3 letters)
        # doesn't pass for an 8-char organ name like "PROSTATE".
        # Use CEILING rounding (math.ceil-style) so "AORTA" (5 chars) requires
        # 4+ letter candidates — otherwise 3-letter noise like "ATA" sneaks through.
        min_cand_chars = max(2, -(-len(organ) * 7 // 10)) if not is_short else 0
        for cand in candidates:
            cand_letter_count = sum(1 for c in cand if not c.isspace())
            if is_short and abs(len(cand) - len(organ)) > 2:
                continue
            if not is_short and cand_letter_count < min_cand_chars:
                continue
            score = SequenceMatcher(None, organ, cand).ratio()
            # Prefix bonus: if first 3 chars match, give a small boost so e.g. "ORAS"
            # picks "ORBITS" over "AORTA" (both score 0.667 without prefix info).
            if len(cand) >= 3 and len(organ) >= 3 and organ[:3] == cand[:3]:
                score += 0.1
            if score > best_score and score >= threshold_for_organ:
                best_score, best_match = score, organ
    return best_match, best_score


def extract_organ_label(img):
    w, h = img.size
    scan_area = img.crop((int(w * 0.15), int(h * 0.10), int(w * 0.90), int(h * 0.90)))

    best_match, best_score, short_match = None, 0.0, None
    for threshold in (STRICT_YELLOW, LOOSE_YELLOW):
        yimg = upscale(isolate_color(scan_area, threshold), 3)
        text = pytesseract.image_to_string(yimg, config="--psm 6").strip()
        if not text:
            continue
        if short_match is None:
            short_match = find_short_code(text)
        match, score = fuzzy_match_organ(text)
        if score > best_score:
            best_score, best_match = score, match
        # SPEED: if STRICT pass already gave high confidence, skip LOOSE pass.
        # The LOOSE pass is a fallback for blurry/dim labels — when STRICT
        # works, running it again is pure waste (saves 1-2s per image on
        # the largest OCR call).
        if best_score >= 0.9:
            break

    if best_score >= ORGAN_FUZZY_THRESHOLD:
        return best_match
    return short_match


# ─── Measurement parsing ────────────────────────────────────────────────────
def parse_value(raw, unit=None, label=None):
    """Parse a value, recovering missing decimals when the magnitude is implausible."""
    raw = raw.strip()
    sign = -1 if raw.startswith("-") else 1
    digits = raw.lstrip("-")

    if "." in digits:
        try:
            return float(raw)
        except ValueError:
            return None
    if not digits.isdigit():
        return None

    val = sign * float(digits)
    is_length = (unit or "").lower() in ("cm", "mm")
    # cm/mm length values >50 are implausible — recover decimal
    if is_length and 3 <= len(digits) <= 4 and abs(val) > 50:
        recovered = digits[:-2] + "." + digits[-2:]
        return sign * float(recovered)
    # Unitless ratio labels (containing "/") with values >10 — recover decimal
    # e.g. "HC/AC 132" → 1.32, "Rt Ut-S/D 413" → 4.13
    if (
        unit is None
        and label is not None
        and "/" in label
        and 3 <= len(digits) <= 4
        and abs(val) >= 10
    ):
        recovered = digits[:-2] + "." + digits[-2:]
        return sign * float(recovered)
    return val


def fix_letter_misreads(label):
    """Replace lowercase 'l' with 'I' when likely an OCR misread of capital I."""
    result = []
    chars = list(label)
    for i, c in enumerate(chars):
        if c == "l":
            prev_upper = i > 0 and chars[i - 1].isupper()
            next_upper = i + 1 < len(chars) and chars[i + 1].isupper()
            at_end = i == len(chars) - 1
            if prev_upper and (next_upper or at_end):
                result.append("I")
                continue
        result.append(c)
    return "".join(result)


# When the yellow annotation OCR is too garbled to identify the organ,
# we can sometimes derive it from a measurement label.  Only used as a
# last-resort fallback so we don't over-claim.
MEASUREMENT_LABEL_TO_ORGAN = {
    "FIB": "TIB/FIB",
    "TIB": "TIB/FIB",
    "HL": "HUMERUS",
}


def derive_organ_from_measurements(measurements):
    for entry in measurements:
        # Take the first whitespace-separated token of the label
        first_token = entry["label"].split()[0].upper().rstrip(",")
        if first_token in MEASUREMENT_LABEL_TO_ORGAN:
            return MEASUREMENT_LABEL_TO_ORGAN[first_token]
    return None


def correct_doppler_label(label):
    """If the label looks Doppler-ish, snap it to the nearest known code."""
    if "-" not in label and "/" not in label:
        return label  # not Doppler-style
    # Generate candidates with letter-misread corrections applied
    candidates = {label, fix_letter_misreads(label)}
    best_score, best_match = 0.0, None
    for code in DOPPLER_CODES:
        for cand in candidates:
            score = SequenceMatcher(None, cand.upper(), code.upper()).ratio()
            if score > best_score:
                best_score, best_match = score, code
    if best_score >= DOPPLER_CORRECT_THRESHOLD:
        return best_match
    return label


def clean_label(text):
    """Strip junk separators and leading non-letter chars from a label string."""
    # Remove trailing junk separators (spaces, equals, tildes, underscores, dashes, em-dashes)
    text = re.sub(r"[\s=~_\-—–]+$", "", text)
    # Remove leading non-letters (e.g. "—— Rt..." → "Rt...", ", Rt..." → "Rt...")
    text = re.sub(r"^[^A-Za-z]+", "", text)
    return text.strip()


def parse_measurement_line(line):
    """Try to extract (label, value, unit, index) from a single line.  Returns None on failure."""
    line = line.strip()
    if not line:
        return None

    # OCR fix: "1 D 0.59cm" sometimes becomes corrupted variants like:
    #   "1 DO. 59cm"   — period/space splits decimal
    #   "1 D0. 59cm"   — zero attached to D
    #   "1 D O 55cm"   — space between D and O
    #   "1 DO 55.0cm"  — Tesseract reads "0.55" as "55.0" (digits + bogus .0)
    #   "1 D055cm"     — fully merged
    #   "1 DO55cm"     — fully merged with O
    # In all cases the real label is "D" and the real value is "0.NN".
    # Patch this back to "1 D 0.NN cm" before parsing so the parser gets
    # the correct label and decimal value.
    line = re.sub(
        # leading-index? + D + (separator)* + (O|0) + (separator)* +
        # 1-3 digits + optional ".0" + cm/mm
        r"^(\s*\d?\s*)D[\s.]*[O0][\s.]*(\d{1,3})(?:\.0)?\s*(cm|mm)\s*$",
        r"\1D 0.\2\3",
        line,
        flags=re.IGNORECASE,
    )

    # OCR fix: Tesseract sometimes reads "9.40cm" as "9 40cm" — splits the
    # decimal at the "." into a space. The number is plausible (9 < 50 for
    # length) so our existing decimal-recovery logic doesn't trigger. We
    # patch this BEFORE the trailing-number match so parsing sees "9.40cm".
    # Match: <single digit> <space(s)> <2 digits> <space(s)>? cm/mm
    # Don't apply if the leading number is multi-digit (could be legit "12 40cm" → 12.40cm? rare).
    # Only fires on N-NN where N is 1-9, NN is 01-99 (typical biometry range).
    line = re.sub(
        r"\b(\d)\s+(\d{2})\s*(cm|mm)\b",
        r"\1.\2\3",
        line,
        flags=re.IGNORECASE,
    )

    # OCR fix: Tesseract reads the digit "4" merged with the preceding decimal
    # point "." as the LETTER "A" when the OCR is on small blurry text. The
    # ".4" glyph (small dot + 4 with crossbar) visually resembles capital "A":
    #   "1 D9A5cm"   ← actually "1 D 9.45cm"  (the ".4" became "A")
    #   "1 D3A6cm"   ← actually "1 D 3.46cm"  (the ".4" became "A")
    # Pattern: digit + "A" + digit, with "A" sitting inside what looks like a
    # number. No legitimate clinical label has this pattern (real labels with
    # "A" never sit between two digits with no spaces).
    # Replace "A" between digits with ".4" to recover the original value.
    line = re.sub(r"(\d)A(\d)", r"\1.4\2", line)

    # Find a number (optionally negative/decimal) followed optionally by a unit,
    # at or near the END of the line.  Allow a stray "/" between number and unit
    # (Tesseract sometimes inserts one).
    m = re.search(rf"(-?\d+(?:\.\d+)?)\s*/?\s*({UNITS})?\s*[^\w]*$", line)
    if not m:
        return None

    value_str = m.group(1)
    unit = m.group(2)
    raw_label = line[: m.start()]

    # Detect leading index BEFORE clean_label strips it.  The OCR text often
    # looks like "2 D" or "1_HL" — we want the digit captured as the index.
    index = None
    idx_m = re.match(r"^\s*(\d)[_\s]+([A-Za-z].*)$", raw_label)
    if idx_m:
        index = int(idx_m.group(1))
        raw_label = idx_m.group(2)

    label_text = clean_label(raw_label)

    if not label_text or not re.search(r"[A-Za-z]", label_text):
        return None

    # Strip leading lowercase-only junk tokens like "il" (OCR misread of "1")
    # so "il HR" becomes "HR".  Real labels start with capital letters or "Rt"/"Lt".
    while True:
        parts = label_text.split(maxsplit=1)
        if len(parts) >= 2 and parts[0].islower() and len(parts[0]) <= 3:
            label_text = parts[1]
        else:
            break

    # Reject when the "label" ends with what looks like a partial number —
    # this happens when OCR fragments one value (e.g. "5.57") into "5.5/" + "7",
    # and our parser would mistakenly treat "D 5.5/" as a label and "7" as the value.
    # EXCEPTION: legitimate volume-axis labels D1/D2/D3/D4 end with a digit but
    # are not partial numbers — they're actual measurement labels.
    if label_text.upper() not in {"D1", "D2", "D3", "D4"} and re.search(r"\d\s*[./]?\s*$", label_text):
        return None

    # Reject labels with garbage letter-then-digit tokens like "WOw2d" or "2ZUW25G".
    # Legitimate digit-bearing tokens (like "20w3d" in "GA 20w3d") always start
    # with a digit, not a letter.
    # EXCEPTION: short volume-axis labels D1/D2/D3/D4 (and case variants) are
    # legitimate — they appear in 3-axis volume measurements
    # ("1 D1 2.92cm, 2 D2 3.08cm, 3 D3 3.69cm, Vol 17.376cm³").
    VOLUME_AXIS_TOKENS = {"D1", "D2", "D3", "D4"}
    for token in label_text.split():
        if not token:
            continue
        if token.upper() in VOLUME_AXIS_TOKENS:
            continue  # legitimate volume-axis label
        if token[0].isalpha() and any(c.isdigit() for c in token):
            return None

    # Reject labels with non-clinical characters (punctuation, symbols).
    # Real labels only contain letters, digits, spaces, hyphens, slashes,
    # underscores, periods, and parentheses (e.g. "Rt Ut-S/D", "GA 20w3d",
    # "FL/AC", "OFD(HC)", "CI(BPD/OFD)").
    if re.search(r"[^A-Za-z0-9\s\-/_.()]", label_text):
        return None

    # Reject labels with a lowercase-letter immediately followed by an
    # uppercase-letter (e.g. "cI" in "PL/I cI/").  Real clinical labels
    # never have this pattern: "Rt Ut-PS" has only upper→lower transitions,
    # "TAmax" only upper→lower, etc.
    if re.search(r"[a-z][A-Z]", label_text):
        return None

    # Junk-filter: require unit, OR a clinical separator pattern (letter-/-letter
    # like "FL/AC" or letter--letter like "Rt Ut-PS"), OR a parenthesized
    # composite like "OFD(HC)", OR a numbered volume axis like "D1"/"D2"/"D3"/"D4".
    # Rejects garbage like "WA 2ZUW25G OY./" where the slash is just at the end
    # of OCR noise.
    if unit is None:
        has_separator = bool(re.search(r"[A-Za-z][\-/][A-Za-z]", label_text))
        has_parens = bool(re.search(r"[A-Za-z]+\([A-Za-z/]+\)", label_text))
        is_volume_axis = label_text.upper() in {"D1", "D2", "D3", "D4"}
        if not (has_separator or has_parens or is_volume_axis):
            return None
    return label_text, value_str, unit, index


def normalize_label(label):
    """Collapse OCR letter ambiguities for dedup."""
    s = label.upper().replace(" ", "")
    return s.replace("L", "I").replace("0", "O")


def extract_measurements_from_text(text):
    results = []
    seen = set()  # (normalized_label, value) tuples

    for raw_line in text.splitlines():
        parsed = parse_measurement_line(raw_line)
        if not parsed:
            continue
        label, value_str, unit, index = parsed

        # Snap Doppler-like labels to known codes
        label = correct_doppler_label(label)

        value = parse_value(value_str, unit, label)
        if value is None:
            continue

        key = (normalize_label(label), value)
        if key in seen:
            continue
        seen.add(key)

        entry = {"label": label, "value": value}
        if unit:
            entry["unit"] = unit.lower()
        if index is not None:
            entry["index"] = index
        results.append(entry)

    return results


def ocr_measurement_crop(crop, inverted=False):
    """OCR a single crop.  When `inverted=True`, invert colors and upscale
    further — useful for small white-on-dark single-line measurement boxes."""

    def fix_slash_seven(t):
        # OCR sometimes reads "7" as "/" inside numbers ("6.73" -> "6./3").
        # Only fires between digits/decimal — leaves clinical labels alone.
        return re.sub(r"(\d\.?)/(?=\d)", r"\g<1>7", t)

    if inverted:
        crop_proc = ImageOps.invert(crop.convert("L"))
        crop_proc = crop_proc.resize(
            (crop.width * 3, crop.height * 3), Image.LANCZOS
        )
    else:
        crop_proc = upscale(crop, 3)
    return fix_slash_seven(pytesseract.image_to_string(crop_proc, config="--psm 6"))


def _extract_measurements_pass(img, crops, inverted):
    """Single full pass over the given crops with the chosen OCR mode.

    Dedup logic across crops within this pass:
    - Exact (label, index, value) duplicate → skip
    - (label, index) slot already locked from a previous crop → skip
      (different value for same slot in a different crop = OCR misread,
       trust the first crop)
    - (label, value) already seen with an *index*, but this entry has no
      index → skip (the second crop saw the same number but lost the
      "1"/"2" prefix; the indexed version is more informative).
      ⚠️ This does NOT skip the *first* unindexed entry for a (label, value)
      that has not yet been seen with an index.  That preserves legitimate
      single-line boxes (e.g. SPLEEN with one D measurement).
    """
    w, h = img.size
    seen_keys = set()
    locked_slots = set()
    indexed_label_values = set()  # (label, value) pairs that arrived WITH an index
    results = []

    for left, top, right, bottom in crops:
        crop = img.crop((int(w * left), int(h * top), int(w * right), int(h * bottom)))
        text = ocr_measurement_crop(crop, inverted=inverted)

        crop_entries = extract_measurements_from_text(text)
        new_slots_this_crop = set()
        for entry in crop_entries:
            label_norm = normalize_label(entry["label"])
            idx = entry.get("index")
            slot = (label_norm, idx)
            key = slot + (entry["value"],)
            if key in seen_keys:
                continue
            if slot in locked_slots:
                continue
            # Cross-crop dedup: if we already have this (label, value) under
            # an index, and this entry has no index, it's a duplicate from
            # a wider crop where OCR missed the "1"/"2" prefix.
            if idx is None and (label_norm, entry["value"]) in indexed_label_values:
                continue
            seen_keys.add(key)
            new_slots_this_crop.add(slot)
            if idx is not None:
                indexed_label_values.add((label_norm, entry["value"]))
            results.append(entry)
        locked_slots |= new_slots_this_crop

    return results


def extract_measurements(img):
    """Run BOTH normal and inverted-color OCR on the standard crops.
    Inversion catches cases where small white-on-dark text has misreads
    (e.g. "." read as "A" in jag5).  Garbage from inverted OCR is filtered
    by the lowercase/uppercase, non-clinical-character, and ratio-decimal
    filters in parse_measurement_line + parse_value.

    SPEED: skip the inverted pass entirely if the normal pass found 2+
    measurements.  Inverted is a fallback for edge cases where normal OCR
    misreads a single character (e.g. jag5: "." → "A" caused only 1 of 2
    measurements to be found).  When normal already returned 2+ values,
    the image is "complete" and inverted would just be wasted work.

    If normal AND inverted both return nothing, fall back to inverted OCR
    on the very tight crop (rukaina3 / rukaina5 single-line boxes)."""
    results = _extract_measurements_pass(img, NORMAL_CROPS, inverted=False)

    # Only run the inverted pass when normal pass might have missed
    # something (count < 2).  This cuts ~50% of OCR calls for typical
    # images where the normal pass fully captures the measurement panel.
    if len(results) < 2:
        inv_results = _extract_measurements_pass(img, NORMAL_CROPS, inverted=True)

        # Merge: keep all from normal, add only NEW entries from inverted.
        # An inverted entry is "new" only if BOTH:
        #   - its (label, index, value) isn't already present, AND
        #   - it doesn't repeat a (label, value) we already have under a
        #     different (or missing) index — that's just OCR losing the index.
        seen_slots = {(normalize_label(e["label"]), e.get("index")) for e in results}
        seen_keys = {(normalize_label(e["label"]), e.get("index"), e["value"]) for e in results}
        seen_label_values = {(normalize_label(e["label"]), e["value"]) for e in results}
        for entry in inv_results:
            label_norm = normalize_label(entry["label"])
            slot = (label_norm, entry.get("index"))
            key = slot + (entry["value"],)
            if key in seen_keys:
                continue
            if slot in seen_slots:
                continue  # different value for same slot — trust normal pass
            if (label_norm, entry["value"]) in seen_label_values:
                continue  # same label+value, different/missing index — duplicate
            seen_slots.add(slot)
            seen_keys.add(key)
            seen_label_values.add((label_norm, entry["value"]))
            results.append(entry)

    if not results:
        # Last-resort fallback: inverted-color on the very tight crop
        results = _extract_measurements_pass(img, INVERTED_FALLBACK_CROPS, inverted=True)
    return results


def process(image_path):
    img = Image.open(image_path)
    measurements = extract_measurements(img)
    organ = extract_organ_label(img)
    # If no organ from yellow text, try to infer one from measurement labels
    # (e.g. a "FIB" measurement implies a TIB/FIB scan).
    if organ is None:
        organ = derive_organ_from_measurements(measurements)
    return {
        "organ": organ,
        "measurements": measurements,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_ultrasound.py <image_path>")
        sys.exit(1)
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)
    print(json.dumps(process(str(image_path)), indent=2))


if __name__ == "__main__":
    main()