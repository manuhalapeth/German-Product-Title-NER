import argparse
import csv
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

# -----------------------------
# Optional category gating
# -----------------------------
try:
    from allowed_by_category import is_allowed  # user-provided helper
except Exception:
    def is_allowed(aspect: str, cat_id: int) -> bool:
        # Fallback: allow everything (you can tighten below if desired)
        return True

# -----------------------------
# Config
# -----------------------------
MIN_LEN = 2
MAX_LEN = 80
# Aggressive duplicate suppression helps precision; keep high but not perfect to allow minor variants
FUZZY_SIM_THRESHOLD = 0.94
# Fβ focus (β>=1 favors recall; we’ll use β=2 by default for safety)
DEFAULT_BETA = 2.0

# Common connectors and pure “O-like” tokens (don’t output as aspect values)
_OISH = {
    "für", "mit", "und", "der", "die", "das", "+", "-", "/", "\\", "|",
    "geeignet", "passend", "inkl", "inkl.", "set", "neu", "original", "or.", "or",
    "von", "zum", "zur", "für:", "mit:", "ohne", "ohne:", "komplett", "kompl.", "satz"
}

# For light unicode normalization without “fixing spelling”
RE_WS = re.compile(r"\s+")
RE_TRIM = re.compile(r"^[\s,.;:/\\|()\[\]{}<>\"'`~]+|[\s,.;:/\\|()\[\]{}<>\"'`~]+$")

# Patterns for key aspects
RE_SAE = re.compile(r"\b(\d{1,2})\s*[Ww]\s*[-/]?\s*(\d{1,2})\b")
RE_DIAM = re.compile(r"(?:Ø|O|DIA\.?|Durchmesser)?\s*0*([1-9]\d{1,3})\s*(?:mm|MM)?\b")
RE_ZAEHNE = re.compile(r"\b(\d{1,3})\b")
RE_YEAR_RANGE = re.compile(r"\b((?:19|20)?\d{2})(?:\s*[-/]\s*((?:19|20)?\d{2}))?\b")
RE_MPN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .\-_/()]*[A-Za-z0-9)]$")

# Value charset guardrail (allow umlauts, ß, basic punct seen in titles)
RE_VALUE_OK = re.compile(r"^[0-9A-Za-zÄÖÜäöüß +\-_/().,*–—°Øø:\[\]]+$")

# Aspects with numeric-leaning normalization
NUMERIC_ASPECTS = {
    "Zähnezahl", "Zahnzahl",
    "Bremsscheiben-Aussendurchmesser", "Durchmesser",
    "Stärke", "Breite", "Länge", "Größe", "Menge"
}

# Aspects where we aggressively collapse spaces/upper (IDs)
ID_ASPECTS = {"Oe/Oem_Referenznummer(N)", "Herstellernummer"}

# Aspects where casing is important (“brands” often uppercase in data)
BRANDY = {"Hersteller", "Produktlinie", "Modell", "Technologie", "Bremsscheibenart", "Oberflächenbeschaffenheit"}

# -----------------------------
# Utilities
# -----------------------------
def similar(a: str, b: str) -> float:
    # Token-sort similarity to catch minor reorderings and punctuation variations
    a_norm = " ".join(sorted(a.lower().split()))
    b_norm = " ".join(sorted(b.lower().split()))
    return SequenceMatcher(None, a_norm, b_norm).ratio()

def is_oish_token(v: str) -> bool:
    w = v.strip().lower()
    return w in _OISH or (len(w) <= 2 and w in {"+", "-", "/", "x"})

def guard_value_charset(v: str) -> bool:
    # Be permissive; only drop truly weird garbage
    return bool(RE_VALUE_OK.match(v))

def clamp_len(v: str) -> bool:
    return MIN_LEN <= len(v) <= MAX_LEN

def clean_basic(v: str) -> str:
    # Normalize dashes, collapse spaces, strip punctuation at ends
    v = v.replace("‐", "-").replace("–", "-").replace("—", "-")
    v = v.replace("’", "'")
    v = RE_WS.sub(" ", v)
    v = RE_TRIM.sub("", v)
    return v

def normalize_value(aspect: str, val: str) -> str or None:
    if not val:
        return None
    v = clean_basic(val)

    if not clamp_len(v):
        return None
    if not guard_value_charset(v):
        return None
    if is_oish_token(v):
        return None

    # Aspect-specific shaping
    if aspect == "SAE_Viskosität":
        m = RE_SAE.search(v)
        if m:
            return f"{int(m.group(1))}W{int(m.group(2))}"
        # keep original if it looks like viscosity (e.g., 0W-40)
        if "w" in v.lower():
            v2 = v.upper().replace(" ", "").replace("/", "").replace("-", "")
            if re.match(r"^\d{1,2}W\d{1,2}$", v2):
                return v2
            return v.upper()
        return None  # unlikely to be valid viscosity

    if aspect in {"Bremsscheiben-Aussendurchmesser", "Durchmesser"}:
        m = RE_DIAM.search(v)
        if m:
            return f"Ø{int(m.group(1))}mm"
        # bare number like 300 or 300MM
        m2 = re.match(r"^0*([1-9]\d{1,3})(?:\s*mm|MM)?$", v)
        if m2:
            return f"Ø{int(m2.group(1))}mm"
        return None

    if aspect in {"Zähnezahl", "Zahnzahl"}:
        m = RE_ZAEHNE.search(v.replace("Zähne", ""))
        if m:
            return m.group(1)
        return None

    if aspect == "Kompatibles_Fahrzeug_Jahr":
        # Keep as provided but normalize common ranges
        m = RE_YEAR_RANGE.search(v)
        if not m:
            return None
        y1, y2 = m.group(1), m.group(2)
        # Expand 2-digit years ambiguously? Keep as-is to avoid hallucinations.
        v_norm = f"{y1}-{y2}" if y2 else y1
        return v_norm

    if aspect in ID_ASPECTS:
        # Tight formatting: remove spaces, keep -,/,_,.,() if inside
        v2 = v.replace(" ", "")
        if RE_MPN.match(v2):
            return v2.upper()
        return None

    if aspect in NUMERIC_ASPECTS:
        # Soft numeric guard: keep values with digits or Ø or mm-like units
        if re.search(r"\d", v) or "Ø" in v or "mm" in v.lower():
            return v.replace("MM", "mm")
        return None

    if aspect in BRANDY:
        # Light casing normalization (don’t fix spelling!)
        # Uppercase short tokens, Title-case multi-token names
        if len(v.split()) == 1 and len(v) <= 6:
            return v.upper()
        return " ".join([w if w.isupper() and len(w) > 3 else (w.upper() if len(w) <= 3 else w.capitalize()) for w in v.split()])

    # Default: return trimmed value
    return v

def resolve_conflicts(aspect: str, values: List[str]) -> List[str]:
    """Prefer longer, more-informative strings and drop substrings/highly-similar duplicates."""
    if not values:
        return values
    # Sort by length desc to prefer richer spans
    values = sorted(set(values), key=lambda s: (-len(s), s))
    kept: List[str] = []
    for v in values:
        if any(similar(v, k) >= FUZZY_SIM_THRESHOLD or (v in k and len(v) <= 6) for k in kept):
            continue
        kept.append(v)
    # For Model vs Product Line ambiguity, keep up to 3 per aspect
    return kept[:3]

def passthrough_allowed(aspect: str, cat_id: int) -> bool:
    try:
        return is_allowed(aspect, cat_id)
    except Exception:
        return True

# -----------------------------
# Core
# -----------------------------
def process_file(input_path: str, output_path: str, beta: float = DEFAULT_BETA) -> Tuple[int, int, int]:
    by_key: Dict[Tuple[int, int], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    malformed = 0
    dropped = 0
    total = 0

    # We avoid pandas to honor “no csv-style quoting” requirement in the output and full control here.
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        for ln, raw in enumerate(f, 1):
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 4:
                malformed += 1
                continue
            rid_s, cid_s, aspect, value = parts
            # Skip header-like lines (rare) or non-integers
            try:
                rid = int(rid_s)
                cid = int(cid_s)
            except ValueError:
                # Probably a header; skip
                continue

            total += 1

            # Drop “O” (Annexure: including O doesn’t change score; dropping reduces noise)
            if aspect.strip() == "O":
                dropped += 1
                continue

            # Enforce category-aspect compatibility
            if not passthrough_allowed(aspect, cid):
                dropped += 1
                continue

            norm = normalize_value(aspect, value)
            if not norm:
                dropped += 1
                continue

            by_key[(rid, cid)][aspect].append(norm)

    # Dedup & conflict resolution
    kept_rows = 0
    with open(output_path, "w", encoding="utf-8", newline="") as out:
        # Per Annexure submission format: literal TABs, no CSV quoting
        for (rid, cid), amap in by_key.items():
            for aspect, vals in amap.items():
                final_vals = resolve_conflicts(aspect, vals)
                for v in final_vals:
                    out.write(f"{rid}\t{cid}\t{aspect}\t{v}\n")
                    kept_rows += 1

    return kept_rows, dropped, malformed

# -----------------------------
# Entry
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Annexure-aligned post-processor for eBay NER predictions")
    ap.add_argument("--input", required=True, help="Path to 4-col TSV (rid, cid, aspect, value)")
    ap.add_argument("--output", required=True, help="Path to write cleaned TSV")
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA, help="F-beta emphasis (>=1 favors recall). Used to tune thresholds if extended.")
    args = ap.parse_args()

    kept, dropped, bad = process_file(args.input, args.output, beta=args.beta)
    print(f"✅ Wrote {kept} cleaned rows → {args.output}")
    if dropped:
        print(f"ℹ️ Dropped {dropped} rows (O-tag, disallowed, or invalid after normalization).", file=sys.stderr)
    if bad:
        print(f"⚠️ Skipped {bad} malformed lines.", file=sys.stderr)

if __name__ == "__main__":
    main()
