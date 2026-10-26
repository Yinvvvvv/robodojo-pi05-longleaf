#!/usr/bin/env bash
#SBATCH --job-name=pi05-five-day-session
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=../logs/session-%j.out
set -euo pipefail
# Catch even a dispatcher crash: this job must not end without human approval.
export WM_ROOT=/hickory/users/y/i/yinxw/WM_Agent
retain_on_exit() {
    code=$?
    trap - EXIT
    echo "SESSION_EXIT_GUARD status=$code; retaining until user-approved release"
    until /usr/bin/python3 "$WM_ROOT/longleaf/train_session.py" --release-approved; do
        sleep 60
    done
    exit "$code"
}
trap retain_on_exit EXIT
source /hickory/users/y/i/yinxw/WM_Agent/longleaf/env.sh
test -f "$WM_ROOT/logs/ENVIRONMENT_READY"
date -Is
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
sha256sum "$WM_ROOT/longleaf/train_session.py" "$WM_ROOT/longleaf/baselines.sbatch"
"$OPENPI_ROOT/.venv/bin/python" "$WM_ROOT/longleaf/train_session.py" "$@"
