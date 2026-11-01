#!/bin/bash
# Full export + measurement pipeline for one checkpoint.
#   usage: [STAGES="1 2 3 4 5 6 7 8"] paper/scripts/final_pipeline.sh CKPT TAG
# Stages: 1 export + residual-format choice (q, q-1, q+1 by the full test gate)   2 rebuild host engine
#         3 golden logits + test suite   4 full test eval (+ npz dump)   5 validation clip eval (fixed engine)
#         6 stream eval, budget-tuned on validation (100 false alarms per background hour)   7 test-stream sweep
#         8 activation ranges.
# Every stage logs to ~/.cache/kws-final/pipeline_TAG/stageN.log and the script stops on the first failure.
# Heavy stages run under a memory cap and wait until MemAvailable >= cap + 2 GiB. Never deletes libmumamba.so
# (make -B), because other jobs on this machine may have it loaded.
set -euo pipefail
CKPT=${1:?checkpoint path}; TAG=${2:?tag}
cd "$(dirname "$0")/../.."
export PYTHONPATH=. PATH=$HOME/miniconda3/envs/kws/bin:$PATH KWS_CKPT=$CKPT
CAPPED=$HOME/.cache/kws-verify-session/capped.sh
LOGDIR=$HOME/.cache/kws-final/pipeline_$TAG; mkdir -p "$LOGDIR"
D=paper/data
STAGES=${STAGES:-"1 2 3 4 5 6 7 8"}

need_mem() {  # need_mem <cap in GiB>: wait (max 2 h) until MemAvailable >= cap + 2
  local need waited=0
  need=$(awk -v c="$1" 'BEGIN {printf "%d", (c + 2) * 1048576}')
  while [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -lt "$need" ]; do
    [ $waited -ge 7200 ] && { echo "timeout waiting for $1G + 2G free memory" >&2; exit 3; }
    sleep 30; waited=$((waited + 30))
  done
}
stage() {  # stage <n> <cap GiB> <command...>
  local n=$1 cap=$2; shift 2
  case " $STAGES " in *" $n "*) ;; *) echo "[stage $n] skipped"; return 0;; esac
  need_mem "$cap"
  echo "[stage $n] $* (cap ${cap}G)"
  if ! $CAPPED ${cap}G "$@" > "$LOGDIR/stage$n.log" 2>&1; then
    echo "[stage $n] FAILED, see $LOGDIR/stage$n.log" >&2; tail -5 "$LOGDIR/stage$n.log" >&2; exit 1
  fi
}

stage 1 3 python paper/scripts/choose_resid_format.py --ckpt "$CKPT" --out $D/resid_format_choice_$TAG.json --work-dir $D/_resid_work_$TAG
stage 2 1 make -C engine -B all
stage 3 3 bash -c "python tests/make_golden.py && pytest -q tests/"
stage 4 2 python -m train.kws.eval_c --ckpt "$CKPT" --engine both --split test --json $D/eval_full_test_$TAG.json --dump-npz $D/logits_$TAG.npz
stage 5 2 python -m train.kws.eval_c --ckpt "$CKPT" --engine fixed --split validation --json $D/eval_val_$TAG.json
stage 6 3.5 python -m train.kws.stream_eval --ckpt "$CKPT" --engine both --tune-on-validation --fa-budget 100 --num-keywords 2000 --json $D/stream_$TAG.json
stage 7 3.5 python -m train.kws.stream_eval --ckpt "$CKPT" --engine both --sweep --num-keywords 2000 --json $D/stream_sweep_$TAG.json
stage 8 3 python paper/scripts/activation_ranges.py $D/activation_ranges_$TAG.json --ckpt "$CKPT"
echo "pipeline $TAG done"
