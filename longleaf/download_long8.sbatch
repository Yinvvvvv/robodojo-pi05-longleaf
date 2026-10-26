#!/usr/bin/env bash
#SBATCH --job-name=pi05-long8-download
#SBATCH --partition=datamover
#SBATCH --array=0-7%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=../logs/long8-download-%A_%a.out
set -euo pipefail
source /hickory/users/y/i/yinxw/WM_Agent/longleaf/env.sh
export HF_XET_NUM_CONCURRENT_RANGE_GETS=4
exec "$OPENPI_ROOT/.venv/bin/python" "$WM_ROOT/longleaf/download_long8.py"
