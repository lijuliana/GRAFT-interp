#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 22.04 GPU machine for the GRAFT toy runs.
# Usage: bash setup_vm.sh   (then reboot once so the NVIDIA driver loads)
set -euo pipefail
exec > >(tee -a "$HOME/setup.log") 2>&1
echo "=== setup start $(date -u +%FT%TZ)"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq nvidia-driver-580-server git build-essential tmux >/dev/null
echo "=== driver packages installed $(date -u +%FT%TZ)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME"
[ -d GRAFT ] || git clone -q https://github.com/jnward/GRAFT
git -C GRAFT checkout -q 6da923dc462342cc55347d8a84c23f5407da4096
[ -d rl-rewardhacking ] || git clone -q https://github.com/ariahw/rl-rewardhacking
[ -d GRAFT-interp ] || git clone -q https://github.com/lijuliana/GRAFT-interp
git -C GRAFT-interp pull -q
cd "$HOME/GRAFT/rl"
uv sync
source .venv/bin/activate
bash vllm_patches/apply.sh
python "$HOME/GRAFT-interp/experiments/cv_routing/make_fold_configs.py" --out configs/cv_routing
cp "$HOME"/GRAFT-interp/experiments/cv_routing/sweep_cv_routing_*.py sweeps/
python -c "import torch, transformers, vllm, trl; print('torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__, 'trl', trl.__version__)"
echo "=== setup done $(date -u +%FT%TZ)"
