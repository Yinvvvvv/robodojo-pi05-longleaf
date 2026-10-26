#!/usr/bin/env bash
# Source on Longleaf; caches and checkpoints belong on Hickory, not NAS home.
export WM_ROOT=/hickory/users/y/i/yinxw/WM_Agent
export XPOLICYLAB_ROOT="$WM_ROOT/src/XPolicyLab"
export OPENPI_ROOT="$XPOLICYLAB_ROOT/policy/Pi_05/openpi"
export PATH="$WM_ROOT/bootstrap/bin:$PATH"
export UV_CACHE_DIR="$WM_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$WM_ROOT/cache/python"
export UV_LINK_MODE=copy
export HF_HOME="$WM_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$WM_ROOT/data/lerobot"
export OPENPI_DATA_HOME="$WM_ROOT/cache/openpi"
export XDG_CACHE_HOME="$WM_ROOT/cache"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export GIT_LFS_SKIP_SMUDGE=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TOKENIZERS_PARALLELISM=false
# Keep Slurm's CUDA_VISIBLE_DEVICES; never replace it with physical GPU IDs.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

