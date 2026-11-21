#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 22.04 GPU machine for the GRAFT toy runs.
# Usage: bash setup_vm.sh   (reboot once afterwards if it had to install the NVIDIA driver)
set -euo pipefail
exec > >(tee -a "$HOME/setup.log") 2>&1
echo "=== setup start $(date -u +%FT%TZ)"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq git build-essential tmux >/dev/null
# Images with a working driver (e.g. AWS deep-learning AMIs) skip this; a plain image needs a reboot afterwards.
if ! nvidia-smi >/dev/null 2>&1; then
  sudo apt-get install -y -qq nvidia-driver-580-server >/dev/null
  echo "=== driver installed: REBOOT before training"
fi
echo "=== driver packages installed $(date -u +%FT%TZ)"
# vLLM's FlashInfer backend compiles a kernel at start-up and needs nvcc, which the driver package lacks.
if [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
  wget -q -O /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i /tmp/cuda-keyring.deb >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq cuda-toolkit-12-8 >/dev/null
fi
grep -q CUDA_HOME "$HOME/.bashrc" || echo 'export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH' >> "$HOME/.bashrc"
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
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
