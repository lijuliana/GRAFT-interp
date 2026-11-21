#!/usr/bin/env bash
# One-time setup of a GPU machine (>= 24 GB) for the no-training Qwen3-8B checks.
# Uses the code, base model and adapters released with Knoche's reproduction
# (samuel500/GRAFT branch loreft, samuel500/graft-qwen3-8b-countdown-sft, samuel500/graft-repro-adapters).
set -euo pipefail
exec > >(tee -a "$HOME/setup.log") 2>&1
echo "=== setup start $(date -u +%FT%TZ)"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq && sudo apt-get install -y -qq git build-essential tmux >/dev/null
# 16 GB of RAM is not enough headroom for a 16.4 GB model load; add swap as a safety net
if [ ! -f /swapfile ]; then sudo fallocate -l 24G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile; fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME"
[ -d GRAFT ] || git clone -q -b loreft https://github.com/samuel500/GRAFT
[ -d rl-rewardhacking ] || git clone -q --depth 1 https://github.com/ariahw/rl-rewardhacking
[ -d GRAFT-interp ] || git clone -q https://github.com/lijuliana/GRAFT-interp
git -C GRAFT-interp pull -q
cd "$HOME/GRAFT/rl"
uv sync
source .venv/bin/activate
bash vllm_patches/apply.sh
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("samuel500/graft-qwen3-8b-countdown-sft", revision="a2d54fdf0aa9fa6ee8e487777ee7d63adaf8ec48",
                      local_dir="/home/ubuntu/models/qwen3-8b-sft")
print("base", p)
runs = ["graft/cdhf100_gr_lccoh64_lr1_s1", "graft/cdhf100_gr_lccoh64_lr3_s9", "graft/cdhf100_gr_lccoh64_lr3_s15",
        "graft-no-routing/cdhf100_noroute_anchor_lr1_s1"]
p = snapshot_download("samuel500/graft-repro-adapters", allow_patterns=[r + "/*" for r in runs],
                      local_dir="/home/ubuntu/adapters")
print("adapters", p)
PY
ls -la "$HOME/models/qwen3-8b-sft" | head; du -sh "$HOME/adapters"
echo "=== setup done $(date -u +%FT%TZ)"
