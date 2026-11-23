#!/usr/bin/env bash
set -euo pipefail
cd /content/physical-ai-practice-2
export PYTHONPATH=/content/ap-physicalai-1
export LD_LIBRARY_PATH=/usr/lib64-nvidia JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl LP_NUM_THREADS=2
mkdir -p console
for name in baseline max_iters_500 box_target_strong qpos_strong learning_rate_3e_minus4; do
    if [[ -f "console/$name.ok" ]]; then
        continue
    fi
    printf '%s Starting %s\n' "$(date -u '+%FT%TZ')" "$name"
    python3 -u train_experiment.py "$name" > "console/$name.log" 2>&1
    if grep -qiE 'CCD overflow|contact buffer overflow|constraint buffer overflow' "console/$name.log"; then
        printf 'Collision buffer overflow in %s\n' "$name" >&2
        exit 2
    fi
    printf '%s Finished %s\n' "$(date -u '+%FT%TZ')" "$name"
    touch "console/$name.ok"
done
