#!/usr/bin/env bash
#SBATCH --job-name=pi05-environment
#SBATCH --partition=datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/setup-%j.out

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
source /hickory/users/y/i/yinxw/WM_Agent/longleaf/env.sh
export JAX_PLATFORMS=cpu
mkdir -p "$WM_ROOT"/{src,cache,data/lerobot,logs,checkpoints}
revision=53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2
date -Is
hostname

if [[ ! -d "$XPOLICYLAB_ROOT/.git" ]]; then
    git init "$XPOLICYLAB_ROOT"
    git -C "$XPOLICYLAB_ROOT" remote add origin https://github.com/XPolicyLab/XPolicyLab.git
    git -C "$XPOLICYLAB_ROOT" fetch --depth=1 origin "$revision"
    git -C "$XPOLICYLAB_ROOT" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$XPOLICYLAB_ROOT" rev-parse HEAD)" == "$revision" ]]

if [[ ! -x "$WM_ROOT/bootstrap/bin/uv" ]]; then
    python3 -m venv "$WM_ROOT/bootstrap"
    "$WM_ROOT/bootstrap/bin/python" -m pip install uv
fi
uv --version
cd "$OPENPI_ROOT"
uv sync --locked --no-dev --group lerobot --python 3.11
# The adapter imports pytest from gemma_pytorch.py even on the JAX path.
uv pip install --python "$OPENPI_ROOT/.venv/bin/python" -e "$XPOLICYLAB_ROOT" 'pytest>=8.3.4,<9'
"$OPENPI_ROOT/.venv/bin/python" - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import os
import jax
import torch
import XPolicyLab
from openpi.training import config
from openpi.training import data_loader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

record = {name: metadata.version(name) for name in
          ['openpi', 'jax', 'jaxlib', 'torch', 'lerobot', 'uv']
          if name != 'uv'}
record['python'] = __import__('sys').version
record['cpu_import_check'] = 'passed'
record['gpu_check'] = 'pending Slurm allocation'
record['config'] = config.get_config('pi05_base_aloha_full_sim_arx-x5_seed_0').name
target = Path(os.environ['WM_ROOT']) / 'logs/environment.json'
target.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
PY
uv pip freeze --python "$OPENPI_ROOT/.venv/bin/python" > "$WM_ROOT/logs/requirements-resolved.txt"
touch "$WM_ROOT/logs/ENVIRONMENT_READY"
echo 'ENVIRONMENT_READY: GPU validation and training data preparation are separate jobs.'
