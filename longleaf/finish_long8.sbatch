#!/usr/bin/env bash
#SBATCH --job-name=pi05-long8-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=../logs/long8-finalize-%j.out
set -euo pipefail
source /hickory/users/y/i/yinxw/WM_Agent/longleaf/env.sh
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
export HF_DATASETS_CACHE="${SLURM_TMPDIR:-/tmp}/long8-finalize-${SLURM_JOB_ID}/datasets"
mkdir -p "$HF_DATASETS_CACHE"
sha256sum "$WM_ROOT/longleaf/prepare_long8.py" "$WM_ROOT/longleaf/finish_long8.py"
exec "$OPENPI_ROOT/.venv/bin/python" "$WM_ROOT/longleaf/finish_long8.py"
