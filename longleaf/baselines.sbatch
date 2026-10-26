#!/usr/bin/env bash
# Real training only: formal datasets and QC must be ready before release.
#SBATCH --job-name=pi05-three-baselines
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=../logs/baselines-%j.out

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch.' >&2; exit 2; }
source /hickory/users/y/i/yinxw/WM_Agent/longleaf/env.sh
# Diagnose runtime allocation: return unused buffers instead of retaining a fragmented BFC pool.
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python="$OPENPI_ROOT/.venv/bin/python"
test -f "$WM_ROOT/logs/ENVIRONMENT_READY"
microbatch_size="$("$python" - "$WM_ROOT/longleaf/session-control-$SLURM_JOB_ID.json" "$SLURM_JOB_ID" <<'PYMICRO'
import json,pathlib,sys
control=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert control['job_id']==sys.argv[2]
print(control.get('global_microbatch_size',4))
PYMICRO
)"
# Explicit human/data acceptance gate. Never fall back to the one-episode pilot.
manifest="$WM_ROOT/data/baseline_manifest.json"
test -f "$manifest" || { echo "Formal data manifest missing: $manifest" >&2; exit 2; }
target_steps="$("$python" - "$WM_ROOT" <<'PYBUDGET'
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(sys.argv[1])/'longleaf'))
from parallel_baselines import training_protocol
print(training_protocol(pathlib.Path(sys.argv[1]))['optimizer_updates'])
PYBUDGET
)"
export HF_HUB_OFFLINE=1
export JAX_COMPILATION_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/pi05-${SLURM_JOB_ID}/jax"
export HF_DATASETS_CACHE="$WM_ROOT/data/hf_datasets_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$HF_DATASETS_CACHE"
cd "$OPENPI_ROOT"
settings=("$@")
if (( ${#settings[@]} == 0 )); then
    settings=(ST-Organize MT-Uniform MT-Organize50)
fi
for setting in "${settings[@]}"; do
    case "$setting" in
        ST-Organize) repo=RoboDojo-organize_table_only-arx_x5-joint ;;
        *) repo=RoboDojo-long_horizon8-arx_x5-joint ;;
    esac
    # A retry must not rerun a completed setting under the active user protocol.
    if "$python" - "$WM_ROOT" "$SLURM_JOB_ID" "$setting" "$manifest" <<'PYDONE'
import pathlib,sys
root,job,setting,manifest = sys.argv[1:]
sys.path.insert(0,str(pathlib.Path(root)/'longleaf'))
from parallel_baselines import owner_complete
if not owner_complete(pathlib.Path(root),job,setting,pathlib.Path(manifest)):
    sys.exit(1)
print('BASELINE_ALREADY_COMPLETE',setting,flush=True)
PYDONE
    then
        continue
    fi
    srun "$python" "$WM_ROOT/longleaf/train_pi05.py" --phase train \
        --mode full --gpus 2 --batch-size 256 --microbatch-size "$microbatch_size" \
        --steps "$target_steps" --seed 0 --exp-name "$setting-seed0" --resume \
        --setting "$setting" --manifest "$manifest" --repo-id "$repo" \
        --save-interval 1000 --log-interval 100 --num-workers 8
done
echo "BASELINES_COMPLETE ${settings[*]}"
