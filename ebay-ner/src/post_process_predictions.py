import argparse
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher

# ===========================================================
# 1. Category-based adaptive thresholding
# ===========================================================
CATEGORY_THRESHOLD_BIAS = {
    1: -0.03,  # loosen for Category 1 (brakes etc.)
    2: +0.02   # tighten for Category 2 (fluids)
}

BASE_THRESHOLD = 0.40
FUZZY_SIM_THRESHOLD = 0.93  # controls dedup merging

# ===========================================================
# 2. Normalization utilities
# ===========================================================
RE_WS = re.compile(r"\s+")
RE_TRIM = re.compile(r"^[\s,.;:/\\|()\[\]{}<>\"'`~]+|[\s,.;:/\\|()\[\]{}<>\"'`~]+$")
RE_MM = re.compile(r"0*([1-9]\d{1,3})\s*(?:mm|MM)?$")
RE_DIAM = re.compile(r"(?:Ø|O|DIA\.?|Durchmesser)?\s*0*([1-9]\d{1,3})\s*(?:mm|MM)?\b")

def normalize_value(v: str) -> str:
    v = v.strip()
    v = v.replace("‐", "-").replace("–", "-").replace("—", "-")
    v = v.replace("’", "'")
    v = RE_WS.sub(" ", v)
    v = RE_TRIM.sub("", v)
    # Normalize diameters and units
    m = RE_DIAM.search(v)
    if m:
        return f"Ø{int(m.group(1))}mm"
    m = RE_MM.match(v)
    if m:
        return f"{int(m.group(1))}mm"
    return v

def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ===========================================================
# 3. Deduplication + subset merging
# ===========================================================
def dedup_and_merge(values):
    """Merge overlapping or similar values; keep the most informative."""
    values = sorted(set(values), key=lambda x: (-len(x), x))
    kept = []
    for v in values:
        if any(similar(v, k) >= FUZZY_SIM_THRESHOLD or v in k for k in kept):
            continue
        kept.append(v)
    return kept

# ===========================================================
# 4. Main processing function
# ===========================================================
def process_file(input_path, output_path):
    data = defaultdict(lambda: defaultdict(list))
    malformed = 0
    dropped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                malformed += 1
                continue
            rid_s, cid_s, aspect, value = parts
            try:
                rid = int(rid_s)
                cid = int(cid_s)
            except ValueError:
                continue

            v_norm = normalize_value(value)
            if not v_norm:
                dropped += 1
                continue
            data[(rid, cid)][aspect].append(v_norm)

    kept = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for (rid, cid), amap in data.items():
            bias = CATEGORY_THRESHOLD_BIAS.get(cid, 0)
            thresh = BASE_THRESHOLD + bias
            for aspect, vals in amap.items():
                merged_vals = dedup_and_merge(vals)
                for v in merged_vals:
                    # Soft filtering: skip if too short or junk-like
                    if len(v) < 2 or len(v) > 80:
                        continue
                    out.write(f"{rid}\t{cid}\t{aspect}\t{v}\n")
                    kept += 1

    print(f" Wrote {kept} cleaned rows → {output_path}")
    print(f" Dropped {dropped} invalid or empty values")
    print(f" Skipped {malformed} malformed lines (if any)")

# ===========================================================
# 5. CLI entrypoint
# ===========================================================
def main():
    ap = argparse.ArgumentParser(description="Post-process predictions for EvalAI submission")
    ap.add_argument("--input", required=True, help="Input 4-col TSV file (record_id, category_id, aspect_name, aspect_value)")
    ap.add_argument("--output", required=True, help="Output cleaned TSV file")
    args = ap.parse_args()

    process_file(args.input, args.output)

if __name__ == "__main__":
    main()
