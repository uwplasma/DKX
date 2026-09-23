#!/bin/bash
# Run a queue of restart_point.py points on a shared host.
#
#   run_queue.sh QUEUE BASE.namelist OUT.jsonl PARALLEL RSS_GIB TIME_CAP
#
# QUEUE holds one point per line: "Nxi Nx restart max_restarts [rss_gib [tol]]",
# the optional columns overriding RSS_GIB and the 1e-10 tolerance. Every point
# is pinned to CORES (default 12-15), uses one BLAS/XLA thread, carries an RSS
# guard and a wall-clock cap, and waits to start until the host has its RSS
# guard plus 1 GiB available. Launch it with setsid so it outlives the shell.
set -u
queue=$1 base=$2 out=$3 parallel=$4 rss=$5 cap=$6
cores=${CORES:-12-15}
here="$(cd "$(dirname "$0")" && pwd)"
py=${PYTHON:-python}
export JAX_ENABLE_X64=True JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

run_one() {
  read -r nxi nx restart maxr guard tol <<<"$1"
  guard=${guard:-$rss} tol=${tol:-1e-10}
  while :; do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge $(( ${guard%.*} + 1 )) ] && break
    echo "$(date +%T) waiting: ${avail} GiB available" >&2; sleep 60
  done
  echo "$(date +%T) start Nxi=$nxi Nx=$nx restart=$restart tol=$tol"
  timeout --signal=KILL "$cap" taskset -c "$cores" "$py" "$here/restart_point.py" \
    "$base" "$out" --nxi "$nxi" --nx "$nx" --restart "$restart" \
    --max-restarts "$maxr" --rss-limit-gib "$guard" --tol "$tol" > /dev/null
  code=$?
  [ $code -eq 137 ] && echo "{\"Nxi\": $nxi, \"Nx\": $nx, \"restart\": $restart, \"status\": \"killed: time cap $cap\"}" >> "$out"
  echo "$(date +%T) end Nxi=$nxi Nx=$nx restart=$restart exit=$code"
}
export -f run_one
export base out rss cap cores here py
grep -v '^\s*#' "$queue" | grep -v '^\s*$' | xargs -P "$parallel" -I{} bash -c 'run_one "{}"'
echo "QUEUE DONE $(date +%T)"
