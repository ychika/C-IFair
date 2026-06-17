#!/usr/bin/env bash
# ============================================================================
# run.sh
# Run main.py in background; write logs under logs/YYMMDD/.
# Example:
#   ./run.sh --dataname linear --seed 23 --N 5000 --epochs 500 --prop_steps 120 --R 8 --lambda 1.0 --d 6
# Options:
#   --python /path/to/python    # choose Python executable
# Logs:
#   logs/YYMMDD/{OUT}-{dataname}_seed{seed}_N{N}_E{EPOCHS}_P{PROP_STEPS}_R{RREP}_lam{LMBDA}_d{D_CLUSTERS}_{YYYYMMDD}_{HHMMSS}.log
# ============================================================================
set -euo pipefail

OUT="test"

DATANAME=linear
#DATANAME=nonlinear
#DATANAME=adult
#DATANAME=german
#DATANAME=oulad
#DATANAME=lin_conn
#DATANAME=lin_inadmissible
SEED=24
N=5000
EPOCHS=1000
PROP_STEPS=500
RREP=8
LMBDA=5.0
D_CLUSTERS=5
PY=python

# parse args
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataname)    DATANAME="$2"; shift 2;;   
    --seed)        SEED="$2"; shift 2;;
    --N)           N="$2"; shift 2;;
    --epochs)      EPOCHS="$2"; shift 2;;
    --prop_steps)  PROP_STEPS="$2"; shift 2;;
    --R)           RREP="$2"; shift 2;;
    --lambda)      LMBDA="$2"; shift 2;;
    --d)           D_CLUSTERS="$2"; shift 2;;
    --python)      PY="$2"; shift 2;;
    *)             ARGS+=("$1"); shift ;;
  esac
done

# make date-based log directory: logs/YYMMDD
DATE8="$(date +%Y%m%d)"   # e.g., 20251121
DATE6="$(date +%y%m%d)"   # e.g., 251121
TIME="$(date +%H%M%S)"
LOGROOT="logs"
LOGDIR="${LOGROOT}/${DATE6}"
mkdir -p "${LOGDIR}"

LOGFILE="${LOGDIR}/${OUT}-${DATANAME}_seed${SEED}_N${N}_E${EPOCHS}_P${PROP_STEPS}_R${RREP}_lam${LMBDA}_d${D_CLUSTERS}_${DATE8}_${TIME}.log"

echo "[INFO] Writing log to ${LOGFILE}"
nohup "${PY}" main.py \
  --dataname "${DATANAME}" \
  --seed "${SEED}" \
  --N "${N}" \
  --epochs "${EPOCHS}" \
  --prop_steps "${PROP_STEPS}" \
  --R "${RREP}" \
  --lambda "${LMBDA}" \
  --d "${D_CLUSTERS}" \
  "${ARGS[@]}" \
  > "${LOGFILE}" 2>&1 &

PID=$!
echo "[INFO] Started PID ${PID}"
echo "[INFO] Tail logs: tail -f ${LOGFILE}"
