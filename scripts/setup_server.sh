#!/usr/bin/env bash
# Set up the CVQ codebase on a Vast.ai (or any CUDA) instance.
# Run from the repository root:  bash scripts/setup_server.sh
set -euo pipefail

echo "==> Python: $(python --version 2>&1)"
python -m pip install --upgrade pip wheel

# Vast.ai images usually ship a CUDA build of torch. Keep it — only install if absent,
# so we don't accidentally replace the GPU build with a CPU one.
if python -c "import torch, torchvision" 2>/dev/null; then
  echo "==> torch already present: $(python -c 'import torch; print(torch.__version__, "cuda", torch.cuda.is_available())')"
else
  echo "==> installing torch + torchvision (default CUDA wheels)"
  python -m pip install "torch>=2.4.1" "torchvision>=0.19.1"
fi

echo "==> installing the rest of the dependencies"
python -m pip install \
  "transformers==4.50.3" \
  "lpips>=0.1.4" \
  "pillow>=10.0.0" \
  "numpy>=1.26,<2.3" \
  "requests>=2.31.0" \
  "tqdm>=4.66.0" \
  "pyyaml>=6.0" \
  "einops>=0.8.0" \
  "tensorboard>=2.16.0"

echo "==> sanity check"
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name, f"{p.total_memory/1e9:.0f}GB")
print("transformers", transformers.__version__)
PY

echo "==> done. Next: bash scripts/run_server.sh"
