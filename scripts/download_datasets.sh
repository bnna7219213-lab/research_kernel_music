#!/usr/bin/env bash
# ============================================================
# T2Music Phase A — Dataset downloader
# Usage: ./download_datasets.sh <musiccaps|fma|all> [--sample N]
# ============================================================
set -euo pipefail

DATASET="${1:?Usage: $0 <musiccaps|fma|all> [--sample N]}"
shift || true

# ---------- defaults ----------
SAMPLE_LIMIT=0   # 0 = all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample) SAMPLE_LIMIT="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ---------- paths (mount on K8s PV /data) ----------
DATA_ROOT="${T2MUSIC_DATA_ROOT:-/data/t2music}"
RAW_DIR="${DATA_ROOT}/raw"
SPEC_DIR="${DATA_ROOT}/_spec"

mkdir -p "${RAW_DIR}/musiccaps" "${RAW_DIR}/fma" "${SPEC_DIR}"

echo ">>> T2Music dataset download | dataset=${DATASET} sample=${SAMPLE_LIMIT:-all}"
echo ">>> DATA_ROOT = ${DATA_ROOT}"

# ============================================================
# MusicCaps (YouTube + annotation CSV)
# ============================================================
download_musiccaps() {
  echo ""
  echo "=== MusicCaps ==="
  echo "Note: MusicCaps audio is hosted on YouTube. Requires yt-dlp."

  # 1. Download annotation CSV
  MC_CSV="${RAW_DIR}/musiccaps/musiccaps.csv"
  if [[ ! -f "${MC_CSV}" ]]; then
    echo ">> Downloading annotation CSV..."
    curl -fsSL "https://raw.githubusercontent.com/nils-wagner/MusicCaps/main/musiccaps.csv" \
        -o "${MC_CSV}" || {
      echo "[WARN] Failed to download MusicCaps CSV from GitHub — retrying alternate mirror"
      curl -fsSL "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps.csv" \
          -o "${MC_CSV}"
    }
  else
    echo ">> Annotation CSV exists, skipping."
  fi

  # 2. Parse YouTube IDs and download audio
  echo ">> Parsing YouTube IDs..."
  MC_LIST="${SPEC_DIR}/musiccaps_ytids.txt"
  python3 - <<PYEOF
import csv, sys
with open("${MC_CSV}") as f:
    reader = csv.DictReader(f)
    ids = [row["ytid"] for row in reader]
limit = ${SAMPLE_LIMIT}
if limit > 0:
    ids = ids[:limit]
with open("${MC_LIST}", "w") as out:
    out.write("\n".join(ids) + "\n")
print(f"Total IDs to download: {len(ids)}")
PYEOF

  if ! command -v yt-dlp >/dev/null 2>&1; then
    echo ">> Installing yt-dlp..."
    python -m pip install --quiet yt-dlp
  fi

  echo ">> Downloading audio via yt-dlp (bestaudio → wav)..."
  cd "${RAW_DIR}/musiccaps"
  yt-dlp --download-archive downloaded.txt \
         -f "bestaudio/best" \
         --output "%(id)s.%(ext)s" \
         --exec "ffmpeg -y -i {} -ac 2 -ar 48000 -sample_fmt s16 {}.wav && rm {}" \
         -a "${MC_LIST}" \
         --no-progress \
         2>&1 | tail -20
  cd -

  COUNT=$(find "${RAW_DIR}/musiccaps" -name "*.wav" | wc -l)
  echo ">> MusicCaps done. WAV files on disk: ${COUNT}"
}

# ============================================================
# FMA medium (archive.org)
# ============================================================
download_fma() {
  echo ""
  echo "=== FMA medium ==="
  echo "Note: FMA audio is hosted on archive.org (~22 GB)."

  FMA_URL="https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
  FMA_ZIP="${RAW_DIR}/fma_medium.zip"

  if [[ ! -f "${FMA_ZIP}" ]]; then
    echo ">> Downloading FMA medium (~22 GB)..."
    wget -q --show-progress -O "${FMA_ZIP}" "${FMA_URL}" || {
      echo "[ERROR] Download failed."
      exit 1
    }
  else
    echo ">> Zip exists, skipping download."
  fi

  if [[ ! -d "${RAW_DIR}/fma_medium" ]]; then
    echo ">> Unzipping..."
    unzip -q -o "${FMA_ZIP}" -d "${RAW_DIR}"
  fi

  # 30s preview clips for codec pretraining
  echo ">> Extracting 30s preview clips (for codec pretraining)..."
  python3 - <<PYEOF
import os, glob, subprocess
src = "${RAW_DIR}/fma_medium"
dst = "${RAW_DIR}/fma/previews_30s"
os.makedirs(dst, exist_ok=True)
limit = ${SAMPLE_LIMIT}
files = sorted(glob.glob(os.path.join(src, "**/*.mp3"), recursive=True))
if limit > 0:
    files = files[:limit]
print(f"FMA medium files total: {len(files)}")
for i, f in enumerate(files):
    basename = os.path.splitext(os.path.basename(f))[0]
    out = os.path.join(dst, f"{basename}.wav")
    if os.path.exists(out):
        continue
    cmd = ["ffmpeg", "-y", "-i", f, "-t", "30", "-ac", "2", "-ar", "48000", "-loglevel", "error", out]
    subprocess.run(cmd, check=False)
    if (i+1) % 100 == 0:
        print(f"  processed {i+1}/{len(files)}")
print("FMA preview extraction done.")
PYEOF

  COUNT=$(find "${RAW_DIR}/fma/previews_30s" -name "*.wav" | wc -l)
  echo ">> FMA medium done. WAV previews on disk: ${COUNT}"
}

# ---------- dispatch ----------
case "${DATASET}" in
  musiccaps)  download_musiccaps ;;
  fma)        download_fma ;;
  all)        download_musiccaps; download_fma ;;
  *)          echo "Unknown dataset: ${DATASET}"; exit 1 ;;
esac

echo ""
echo "=== All done. Data root: ${DATA_ROOT} ==="
