#!/usr/bin/env bash
# setup_mac.sh
# macOS setup using pip and venv (Apple Silicon friendly best-effort)
# Usage: chmod +x setup_mac.sh && ./setup_mac.sh

set -euo pipefail

echo "\n=== Smart Traffic - macOS Setup (pip + venv) ===\n"

# 1) Create & activate virtual environment
VENV_DIR=".venv_mac"https://github.com/Pavan67u/smart_traffic
if [ -d "$VENV_DIR" ]; then
  echo "Virtual env '$VENV_DIR' already exists. Skipping creation." 
else
  echo "Creating virtual environment at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# 2) Upgrade pip/wheel/setuptools
echo "Upgrading pip, setuptools and wheel..."
python -m pip install --upgrade pip setuptools wheel

# 3) Try installing PyTorch (best-effort via pip)
# Note: On Apple Silicon (MPS), conda/miniforge is recommended for reliable MPS-enabled builds.
# This script attempts a pip install; if it fails we will continue but warn the user.

echo "\nAttempting to install PyTorch (pip). For best MPS support use Miniforge/conda if this fails."
PYTORCH_OK=0
if python -c "import sys,platform; print(platform.system(), platform.machine());" | grep -qi Darwin; then
  echo "Detected macOS. Trying pip install for PyTorch wheels (may be CPU-only)."
fi

# Try official PyTorch pip index (CPU wheels) — works as fallback
if python -c "import torch" 2>/dev/null; then
  echo "PyTorch already installed."; PYTORCH_OK=1
else
  set +e
  python -m pip install --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu
  RC=$?
  set -e
  if [ "$RC" -eq 0 ]; then
    echo "PyTorch installed via pip (CPU build)."
    PYTORCH_OK=1
  else
    echo "\nWarning: pip installation of PyTorch failed or only CPU wheels available."
    echo "If you need Apple MPS (GPU) support, please install Miniforge and install PyTorch from conda-forge following https://pytorch.org."
  fi
fi

# 4) Install project requirements (if any)
if [ -f infra/requirements.txt ]; then
  echo "\nInstalling packages from infra/requirements.txt..."
  # allow failures of single packages while continuing
  python -m pip install -r infra/requirements.txt || echo "Some packages failed to install from requirements.txt; check the output above." 
else
  echo "No infra/requirements.txt found — skipping."
fi

# 5) Install core project packages
echo "\nInstalling core packages: ultralytics, opencv-python, labelImg..."
python -m pip install -U ultralytics opencv-python labelImg || echo "Some core packages failed to install. Check pip output above." 

# 6) Final notes
echo "\nSetup complete (venv: $VENV_DIR)."
cat <<'EOF'
Next steps (on mac terminal):

# Activate the environment
source .venv_mac/bin/activate

# Verify PyTorch and device support (in Python REPL)
python - <<PY
import torch
print('torch:', torch.__version__)
try:
    print('mps available:', torch.backends.mps.is_available())
except Exception as e:
    print('mps check error:', e)
PY

# Run training (use 'mps' device if supported, otherwise 'cpu')
# Replace epochs/batch as needed
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='training/yolo/data_vehicles.yaml', epochs=10, imgsz=640, batch=8, device='mps')"

# If the above fails, try device='cpu' or use colab/gpu machine for faster training.
EOF

exit 0
