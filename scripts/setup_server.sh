#!/usr/bin/env bash
# Set up the CVQ codebase on a Vast.ai (or any CUDA) instance.
# Run from the repository root:  bash scripts/setup_server.sh
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  echo "==> using uv to build .venv (python 3.12)"
  uv venv --python 3.12 .venv
  # On Linux/x86_64 the default torch wheel is the CUDA build — no extra index needed.
  uv pip install -e .
else
  echo "==> uv not found; falling back to python3 venv + pip"
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip wheel
  ./.venv/bin/python -m pip install -e .
fi

echo "==> sanity check"
./.venv/bin/python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name, f"{p.total_memory/1e9:.0f}GB")
print("transformers", transformers.__version__)
PY

echo "==> done. Next: bash scripts/run_server.sh"
