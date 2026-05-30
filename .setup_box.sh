#!/bin/bash
# Box-side setup: create uv venv, install CUDA torch stack + project deps.
# Runs ON THE BOX. Logs to /root/ImageLab/setup.log.
set -uo pipefail
cd /root/ImageLab || exit 9
export UV_HTTP_TIMEOUT=600
echo "=== uv venv ==="
uv venv --python 3.12 .venv 2>&1 | tail -3
. .venv/bin/activate
echo "py: $(which python) $(python --version)"
echo "=== install torch (cu130) ==="
uv pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision 2>&1 | tail -5
echo "=== install project deps ==="
uv pip install transformers lpips pillow "numpy>=1.26,<2.3" requests tqdm pyyaml einops tensorboard wandb torchmetrics 2>&1 | tail -5
echo "=== verify ==="
python -c "import torch;print('TORCH',torch.__version__,'cuda',torch.cuda.is_available());print('DEV',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')" 2>&1 | tail -4
python -c "import transformers,torchvision,lpips,wandb,einops,torchmetrics;print('DEPS_OK',transformers.__version__,torchvision.__version__)" 2>&1 | tail -2
echo "=== SETUP_DONE ==="
