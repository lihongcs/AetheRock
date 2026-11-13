#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_NAME=$(basename "$0" .sh)
exec > >(tee -a "logs/${SCRIPT_NAME}_${TIMESTAMP}.log") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG_NAME="${CONFIG_NAME:-pi05_bread_moving_health_distill_tactile_wrist_force}"
EXP_NAME="${EXP_NAME:-501_bread_moving_multihealth_pi05_health_distill_tactile_wrist_force}"
export WANDB_DIR="${WANDB_DIR:-checkpoints/${CONFIG_NAME}/${EXP_NAME}}"

# Relative to the openpi repo root after the cd above.
DATA_ROOT_BASE="${DATA_ROOT_BASE:-Data/501_bread_moving_lerobot_multihealth}"
HEALTH0_DATA_ROOT="${HEALTH0_DATA_ROOT:-${DATA_ROOT_BASE}/501_bread_moving_lerobot_health0}"
HEALTH50_DATA_ROOT="${HEALTH50_DATA_ROOT:-${DATA_ROOT_BASE}/501_bread_moving_lerobot_health50}"
HEALTH100_DATA_ROOT="${HEALTH100_DATA_ROOT:-${DATA_ROOT_BASE}/501_bread_moving_lerobot_health100}"

HEALTH_DATA_ROOTS=(
  "${HEALTH0_DATA_ROOT}"
  "${HEALTH50_DATA_ROOT}"
  "${HEALTH100_DATA_ROOT}"
)
HEALTH_REPO_IDS=(
  "501_bread_moving_lerobot_health0"
  "501_bread_moving_lerobot_health50"
  "501_bread_moving_lerobot_health100"
)
HEALTH_LABELS=("0" "50" "100")

BATCH_SIZE="${BATCH_SIZE:-144}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
# This workload is video-decode bound; the latest profile starves the GPUs with
# 16 workers, while 32 workers keeps Data avg much closer to model time.
NUM_WORKERS="${NUM_WORKERS:-48}"
# H200 generally has enough memory to prefer data parallel groups over full
# 4-GPU FSDP for throughput. Override to FSDP_DEVICES=4 if this OOMs.
FSDP_DEVICES="${FSDP_DEVICES:-1}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OPENPI_DATALOADER_PREFETCH_FACTOR="${OPENPI_DATALOADER_PREFETCH_FACTOR:-2}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_force_compilation_parallelism=16}"

echo "=========================================="
echo "Starting OpenPI PI05 multi-health distillation training"
echo "Config: ${CONFIG_NAME}"
echo "Experiment: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "FSDP devices: ${FSDP_DEVICES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Train steps: ${NUM_TRAIN_STEPS}"
echo "Save interval: ${SAVE_INTERVAL}"
echo "Num workers: ${NUM_WORKERS}"
echo "DataLoader prefetch factor: ${OPENPI_DATALOADER_PREFETCH_FACTOR}"
echo "OMP/MKL/OPENBLAS/NUMEXPR threads: ${OMP_NUM_THREADS}/${MKL_NUM_THREADS}/${OPENBLAS_NUM_THREADS}/${NUMEXPR_NUM_THREADS}"
echo "Health data roots:"
for data_root in "${HEALTH_DATA_ROOTS[@]}"; do
  echo "  - ${data_root}"
done
echo "Resume: ${RESUME}"
echo "Overwrite: ${OVERWRITE}"
echo "=========================================="

for data_root in "${HEALTH_DATA_ROOTS[@]}"; do
  if [[ ! -d "${data_root}/data" || ! -d "${data_root}/meta" || ! -d "${data_root}/videos" ]]; then
    echo "ERROR: Expected LeRobot dataset folders data/meta/videos under: ${data_root}"
    exit 1
  fi
done

CHECKPOINT_DIR="checkpoints/${CONFIG_NAME}/${EXP_NAME}"
LATEST_CKPT="$(
  find "${CHECKPOINT_DIR}" -maxdepth 1 -mindepth 1 -type d -name '[0-9]*' 2>/dev/null \
    | awk -F/ '{print $NF}' \
    | sort -n \
    | tail -n 1 \
    || true
)"

if [[ "${RESUME}" == "1" && "${OVERWRITE}" == "1" ]]; then
  echo "ERROR: RESUME=1 and OVERWRITE=1 cannot be used together."
  exit 1
fi

RUN_MODE_FLAGS=()
if [[ "${RESUME}" == "1" ]]; then
  if [[ -z "${LATEST_CKPT}" ]]; then
    echo "ERROR: RESUME=1 but no numeric checkpoint was found under ${CHECKPOINT_DIR}."
    echo "Refusing to start from scratch silently."
    exit 1
  fi
  echo "Resuming from checkpoint step: ${LATEST_CKPT}"
  RUN_MODE_FLAGS=(--resume)
elif [[ "${OVERWRITE}" == "1" ]]; then
  echo "WARNING: OVERWRITE=1 will delete any existing checkpoint directory: ${CHECKPOINT_DIR}"
  RUN_MODE_FLAGS=(--overwrite)
elif [[ -d "${CHECKPOINT_DIR}" ]]; then
  echo "ERROR: Checkpoint directory already exists: ${CHECKPOINT_DIR}"
  echo "Use RESUME=1 to continue from an existing checkpoint, or OVERWRITE=1 to intentionally start over."
  exit 1
fi

MASK_FLAGS=()
if [[ -n "${TACTILE_MASK_RATIO:-}" ]]; then
  MASK_FLAGS=(
    --data.use-tactile-mask
    --data.tactile-mask-ratio "${TACTILE_MASK_RATIO}"
    --data.batch-mask-ratio "${BATCH_MASK_RATIO:-0.5}"
  )
fi

"${PYTHON_BIN}" scripts/compute_norm_stats.py \
  --config-name "${CONFIG_NAME}" \
  --health-data-roots "${HEALTH_DATA_ROOTS[@]}" \
  --health-labels "${HEALTH_LABELS[@]}" \
  --health-repo-ids "${HEALTH_REPO_IDS[@]}"

"${PYTHON_BIN}" scripts/train.py \
  "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --num-workers "${NUM_WORKERS}" \
  --fsdp-devices "${FSDP_DEVICES}" \
  --data.health-data-roots "${HEALTH_DATA_ROOTS[@]}" \
  --data.health-repo-ids "${HEALTH_REPO_IDS[@]}" \
  --data.health-labels "${HEALTH_LABELS[@]}" \
  ${MASK_FLAGS[@]+"${MASK_FLAGS[@]}"} \
  ${RUN_MODE_FLAGS[@]+"${RUN_MODE_FLAGS[@]}"}
