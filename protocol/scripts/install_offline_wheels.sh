#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

WHEEL_DIR="${PROJECT_ROOT}/offline_wheels"
LOCK_FILE="${PROJECT_ROOT}/environment/offline-wheel-lock.txt"

if [[ ! -d "${WHEEL_DIR}" ]]; then
  echo "Offline wheel directory not found: ${WHEEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${WHEEL_DIR}/SHA256SUMS" ]]; then
  echo "Wheel checksum manifest not found: ${WHEEL_DIR}/SHA256SUMS" >&2
  exit 1
fi

echo "[1/5] Platform check"
python - <<'PY'
import platform, sys
if sys.version_info[:2] != (3, 10):
    raise RuntimeError(f"Expected Python 3.10, found {sys.version}")
if platform.machine() not in {"x86_64", "AMD64"}:
    raise RuntimeError(f"Expected x86_64, found {platform.machine()}")
print(sys.version)
print(platform.platform())
PY

echo "[2/5] Verify every copied wheel"
(
  cd "${WHEEL_DIR}"
  sha256sum --check SHA256SUMS
)

echo "[3/5] Verify image-provided core stack; it will not be replaced"
python - <<'PY'
import numpy, PIL, scipy, pandas, torch, torchvision, yaml
expected = {
    "torch": (torch.__version__, "2.3.0+cu121"),
    "torchvision": (torchvision.__version__, "0.18.0+cu121"),
    "numpy": (numpy.__version__, "1.26.3"),
    "Pillow": (PIL.__version__, "10.2.0"),
    "scipy": (scipy.__version__, "1.15.3"),
    "pandas": (pandas.__version__, "2.3.3"),
    "PyYAML": (yaml.__version__, "6.0.3"),
}
for name, (actual, wanted) in expected.items():
    if actual != wanted:
        raise RuntimeError(f"{name}: expected {wanted}, found {actual}")
    print(f"  {name}={actual}: OK")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available to PyTorch")
print(f"  CUDA runtime={torch.version.cuda}; visible GPUs={torch.cuda.device_count()}")
PY

echo "[4/5] Install locked offline packages into the current user's site-packages"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
python -m pip install \
  --user \
  --no-index \
  --no-deps \
  --upgrade \
  --find-links "${WHEEL_DIR}" \
  --requirement "${LOCK_FILE}"

echo "[5/5] Import audit"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  python scripts/check_offline_environment.py

echo "OK: offline VARSR dependencies installed and audited."
echo "Next command:"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_4gpu.sh --mode synthetic_cfg --negative-fraction 0.02 --resume auto"
