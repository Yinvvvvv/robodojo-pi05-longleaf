#!/usr/bin/env bash
#SBATCH --job-name=pi05-long8-convert
#SBATCH --array=0-7%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=../logs/long8-convert-%A_%a.out
set -euo pipefail
source /hickory/users/y/i/yinxw/WM_Agent/longleaf/env.sh
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="${SLURM_TMPDIR:-/tmp}/long8-${SLURM_JOB_ID}/datasets"
export OMP_NUM_THREADS=1
mkdir -p "$HF_DATASETS_CACHE"
sha256sum "$WM_ROOT/longleaf/prepare_long8.py" "$WM_ROOT/longleaf/download_long8.py"
exec "$OPENPI_ROOT/.venv/bin/python" "$WM_ROOT/longleaf/prepare_long8.py" "$@"
