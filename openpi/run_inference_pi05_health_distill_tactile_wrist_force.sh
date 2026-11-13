#!/bin/bash
# Generic PI05 real-robot inference for health-distill + wrist + tactile + force prediction.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# User configuration
# ==============================================================================
# Wrist camera is used by the policy. RealSense and tactile cameras are recorded
# with wrist by the inference client.
CAMERA_CONFIG_PATH="${CAMERA_CONFIG_PATH:-src/perception/configs/camera/handcap_realsense_tactile.json}"

# Optional external overview camera. Leave empty to use only internal wrist+RealSense
# recording.
export RECORDING_INDEX="${RECORDING_INDEX:-}"
# 3. MVS 相机信息增益 (Gain)
# 可调节范围通常在 0.0 到 20.0 之间。调大画面变亮但噪点增加，调小画面变暗但更干净。
export MVS_CAMERA_GAIN="${MVS_CAMERA_GAIN:-15.0}"

# Tactile cameras are defined in handcap_realsense_tactile.json. Set these env
# vars only when you intentionally want to override the config for a one-off run.
TACTILE_CONFIG_OVERRIDE_ARGS=()
if [[ -n "${TACTILE_LEFT_INDEX+x}" ]]; then
  TACTILE_CONFIG_OVERRIDE_ARGS+=(--left-video-index "${TACTILE_LEFT_INDEX}")
fi
if [[ -n "${TACTILE_RIGHT_INDEX+x}" ]]; then
  TACTILE_CONFIG_OVERRIDE_ARGS+=(--right-video-index "${TACTILE_RIGHT_INDEX}")
fi
if [[ -n "${TACTILE_WIDTH+x}" ]]; then
  TACTILE_CONFIG_OVERRIDE_ARGS+=(--tactile-width "${TACTILE_WIDTH}")
fi
if [[ -n "${TACTILE_HEIGHT+x}" ]]; then
  TACTILE_CONFIG_OVERRIDE_ARGS+=(--tactile-height "${TACTILE_HEIGHT}")
fi
if [[ -n "${TACTILE_CAPTURE_WIDTH+x}" ]]; then
  TACTILE_CONFIG_OVERRIDE_ARGS+=(--tactile-capture-width "${TACTILE_CAPTURE_WIDTH}")
fi
if [[ -n "${TACTILE_CAPTURE_HEIGHT+x}" ]]; then
  TACTILE_CONFIG_OVERRIDE_ARGS+=(--tactile-capture-height "${TACTILE_CAPTURE_HEIGHT}")
fi
export TACTILE_CALIBRATION="${TACTILE_CALIBRATION:-1}"
TACTILE_CALIBRATION_ATTEMPTS="${TACTILE_CALIBRATION_ATTEMPTS:-120}"





POLICY_CONFIG="${POLICY_CONFIG:-pi05_512_close_bottle_health_distill_tactile_wrist_force}"
POLICY_DIR="${POLICY_DIR:-}"
PROMPT="${PROMPT:-}"

if [[ -z "${POLICY_CONFIG}" || -z "${POLICY_DIR}" || -z "${PROMPT}" ]]; then
  echo "ERROR: POLICY_CONFIG, POLICY_DIR, and PROMPT are required."
  echo ""
  echo "Example:"
  echo "  POLICY_CONFIG=pi05_516_stowing_health_distill_tactile_wrist_force \\"
  echo "  POLICY_DIR=checkpoints/pi05_516_stowing_health_distill_tactile_wrist_force/516_stowing_multihealth_pi05_health_distill_tactile_wrist_force/100000 \\"
  echo "  PROMPT='Sequentially move the yellow rectangular block, the orange box, and the yellow trapezoidal block into the paper box.' \\"
  echo "  bash run_inference_pi05_health_distill_tactile_wrist_force.sh"
  exit 1
fi

case "${POLICY_CONFIG}" in
  pi05_stowing_490_distill_tactile_wrist_force|pi05_stowing_distill_tactile_wrist_force)
    echo "Normalizing legacy POLICY_CONFIG=${POLICY_CONFIG} to pi05_516_stowing_health_distill_tactile_wrist_force"
    POLICY_CONFIG="pi05_516_stowing_health_distill_tactile_wrist_force"
    ;;
  pi05_513_screw_490_distill_tactile_wrist_force|pi05_513_screw_distill_tactile_wrist_force)
    echo "Normalizing legacy POLICY_CONFIG=${POLICY_CONFIG} to pi05_513_screw_health_distill_tactile_wrist_force"
    POLICY_CONFIG="pi05_513_screw_health_distill_tactile_wrist_force"
    ;;
  pi05_512_close_bottle_490_distill_tactile_wrist_force|pi05_512_close_bottle_distill_tactile_wrist_force)
    echo "Normalizing legacy POLICY_CONFIG=${POLICY_CONFIG} to pi05_512_close_bottle_health_distill_tactile_wrist_force"
    POLICY_CONFIG="pi05_512_close_bottle_health_distill_tactile_wrist_force"
    ;;
esac

export POLICY_CONFIG
export POLICY_DIR
export PROMPT
export TASK_NAME="${TASK_NAME:-${POLICY_CONFIG#pi05_}}"

# Match train_health_distill_tactile_wrist_force.sh:
#   use_tactile=True, force_predict=True.
export FORCE_PREDICT="${FORCE_PREDICT:-true}"
export FORCE_GUIDE="${FORCE_GUIDE:-false}"
export INFER_FORCE_CONTROL="${INFER_FORCE_CONTROL:-false}"

if [[ -z "${NORM_ASSET_ID:-}" ]]; then
  CONFIG_BODY="${POLICY_CONFIG#pi05_}"
  HEALTH_SUFFIX="_health_distill_tactile_wrist_force"
  if [[ "${CONFIG_BODY}" == *"${HEALTH_SUFFIX}" ]]; then
    export NORM_ASSET_ID="${CONFIG_BODY%"${HEALTH_SUFFIX}"}_health_distill"
  else
    echo "ERROR: This generic script is only for health-distill tactile+wrist+force models."
    echo "Got POLICY_CONFIG=${POLICY_CONFIG}"
    echo ""
    echo "Use a config like:"
    echo "  pi05_513_screw_health_distill_tactile_wrist_force"
    echo "  pi05_516_stowing_health_distill_tactile_wrist_force"
    echo "  pi05_512_close_bottle_health_distill_tactile_wrist_force"
    echo ""
    echo "Or set NORM_ASSET_ID explicitly only if you know this checkpoint uses a matching health-distill asset."
    exit 1
  fi
fi

SKIP_NORM_STATS_CHECK="${SKIP_NORM_STATS_CHECK:-0}"
if [[ "${SKIP_NORM_STATS_CHECK}" != "1" && "${POLICY_DIR}" != gs://* && "${POLICY_DIR}" != s3://* ]]; then
  NORM_STATS_PATH="${POLICY_DIR}/assets/${NORM_ASSET_ID}/norm_stats.json"
  if [[ ! -f "${NORM_STATS_PATH}" ]]; then
    echo "ERROR: Norm stats not found for this health-distill policy."
    echo "Expected: ${NORM_STATS_PATH}"
    echo ""
    echo "This script uses POLICY_CONFIG=${POLICY_CONFIG}, whose asset id is ${NORM_ASSET_ID}."
    echo "Make sure the exported checkpoint keeps the training checkpoint's assets/ directory,"
    echo "or copy norm_stats.json into the expected path above."
    exit 1
  fi
fi



# ==============================================================================

if [[ -z "${FLEXIV_ROBOT_IP:-}" || -z "${FLEXIV_ROBOT_SN:-}" || -z "${FLEXIV_INIT_POSE:-}" ]]; then
  echo "Set FLEXIV_ROBOT_IP, FLEXIV_ROBOT_SN, and the verified FLEXIV_INIT_POSE for your setup." >&2
  exit 1
fi

RECORD_PID=""
RECORD_FILE=""
RECORD_ROOT="${RECORD_ROOT:-real_robot_inference_recording}"
if [[ "$RECORD_ROOT" == /* ]]; then
  RECORD_ROOT_PATH="$RECORD_ROOT"
else
  RECORD_ROOT_PATH="${SCRIPT_DIR}/${RECORD_ROOT}"
fi
CLEANED_UP=0
cleanup_recording() {
  if [[ "$CLEANED_UP" == "1" ]]; then
    return
  fi
  CLEANED_UP=1

  if [[ -n "$RECORD_PID" ]]; then
    echo "Stopping external recording process $RECORD_PID..."
    kill -TERM "$RECORD_PID" 2>/dev/null || true
    wait "$RECORD_PID" 2>/dev/null || true
  fi

  sleep 1.5

  if [[ ! ( -n "$RECORD_FILE" && -f "$RECORD_FILE" ) ]]; then
    return
  fi

  echo ""
  read -p "❓ 外置相机整段录像是否需要保留? 输入 y 保留，直接回车或其他键删除 [y/N]: " keep_data </dev/tty
  case "$keep_data" in
    y|Y|yes|Yes )
      echo "✅ 已保留外置录像。"
      SPEEDUP_SCRIPT="${SCRIPT_DIR}/../speedup_video.py"
      if [[ -f "$SPEEDUP_SCRIPT" ]]; then
        echo "Applying 3x speedup to $RECORD_FILE..."
        python "$SPEEDUP_SCRIPT" "$RECORD_FILE" --speed 3
      else
        echo "Skipping 3x speedup because $SPEEDUP_SCRIPT was not found."
      fi
      ;;
    * )
      echo "🗑️  不保留，正在清理外置录像..."
      rm -f "$RECORD_FILE"
      echo "已删除外部录像: $RECORD_FILE"
      ;;
  esac
}
trap cleanup_recording EXIT INT TERM

if [[ -n "$RECORDING_INDEX" ]]; then
  mkdir -p "recordings/${POLICY_CONFIG}"
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  RECORD_FILE="recordings/${POLICY_CONFIG}/${TIMESTAMP}.mp4"
  echo "Starting background recording on external camera $RECORDING_INDEX to $RECORD_FILE"
  python scripts/record_video.py "$RECORDING_INDEX" "$RECORD_FILE" &
  RECORD_PID=$!
fi

export FLEXIV_LOCAL_IP="${FLEXIV_LOCAL_IP:-}"
export FLEXIV_GRIPPER_NAME="${FLEXIV_GRIPPER_NAME:-Flexiv-GN01}"
export FLEXIV_ACTION_FRAME="${FLEXIV_ACTION_FRAME:-flange}"
export FLEXIV_USE_DIRECT_EEF="${FLEXIV_USE_DIRECT_EEF:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export FLEXIV_ARM_MAX_LINEAR_VEL="${FLEXIV_ARM_MAX_LINEAR_VEL:-1.0}"
export FLEXIV_ARM_MAX_ANGULAR_VEL="${FLEXIV_ARM_MAX_ANGULAR_VEL:-1.0}"
export FLEXIV_ARM_MAX_LINEAR_ACC="${FLEXIV_ARM_MAX_LINEAR_ACC:-1.0}"
export FLEXIV_ARM_MAX_ANGULAR_ACC="${FLEXIV_ARM_MAX_ANGULAR_ACC:-1.0}"
export FLEXIV_GRIPPER_MOVE_VELOCITY="${FLEXIV_GRIPPER_MOVE_VELOCITY:-0.1}"
export FLEXIV_GRIPPER_MOVE_FORCE="${FLEXIV_GRIPPER_MOVE_FORCE:-20}"
export FLEXIV_ENABLE_SAFETY_CLIP="${FLEXIV_ENABLE_SAFETY_CLIP:-1}"
RECORD_HZ="${RECORD_HZ:-20}"

TACTILE_CALIBRATION_FLAG=()
case "${TACTILE_CALIBRATION}" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    TACTILE_CALIBRATION_FLAG=(--no-tactile-calibration)
    ;;
  *)
    TACTILE_CALIBRATION_FLAG=(--tactile-calibration)
    ;;
esac

EXTRA_REMOTE_INFERENCE_ARGS=()
if [[ "${FORCE_PREDICT:-false}" == "true" ]]; then
  EXTRA_REMOTE_INFERENCE_ARGS+=(--force-predict)
fi
if [[ "${FORCE_GUIDE:-false}" == "true" ]]; then
  EXTRA_REMOTE_INFERENCE_ARGS+=(--force-guide)
fi
if [[ "${INFER_FORCE_CONTROL:-false}" == "true" ]]; then
  EXTRA_REMOTE_INFERENCE_ARGS+=(--infer-force-control)
fi

echo "Flexiv Cartesian EEF control:"
echo "  action frame: ${FLEXIV_ACTION_FRAME}"
echo "  linear vel/acc:  ${FLEXIV_ARM_MAX_LINEAR_VEL} / ${FLEXIV_ARM_MAX_LINEAR_ACC}"
echo "  angular vel/acc: ${FLEXIV_ARM_MAX_ANGULAR_VEL} / ${FLEXIV_ARM_MAX_ANGULAR_ACC}"
echo "  gripper vel/force: ${FLEXIV_GRIPPER_MOVE_VELOCITY} / ${FLEXIV_GRIPPER_MOVE_FORCE}"
echo "  safety clip: ${FLEXIV_ENABLE_SAFETY_CLIP}"
echo "  camera config: ${CAMERA_CONFIG_PATH}"
echo -e "\033[31m  MVS camera gain: ${MVS_CAMERA_GAIN}\033[0m"
echo -e "\033[31m  POLICY_CONFIG: ${POLICY_CONFIG}\033[0m"
echo -e "\033[31m  POLICY_DIR: ${POLICY_DIR}\033[0m"
echo -e "\033[31m  PROMPT: ${PROMPT}\033[0m"
echo "  tactile config: static handcap_realsense_tactile.json unless env override is set"
echo "  tactile calibration: ${TACTILE_CALIBRATION} (attempts=${TACTILE_CALIBRATION_ATTEMPTS})"
echo "  record hz: ${RECORD_HZ}"
echo "  record root: ${RECORD_ROOT_PATH}"
echo "  main recording: realsense | wrist | calibrated_left_tactile | calibrated_right_tactile"
echo "Runtime controls: SPACE=start/stop one episode, r=reset to FLEXIV_INIT_POSE, Ctrl+C=exit session"

CTRL_FREQ="${CTRL_FREQ:-10}"
STEPS_PER_INFERENCE="${STEPS_PER_INFERENCE:-20}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACTION_LATENCY="${ACTION_LATENCY:-0.0}"
TASK_NAME="${TASK_NAME:-handcap_flexiv_mvs_metrics}"
METRIC_MONITOR_HZ="${METRIC_MONITOR_HZ:-50}"
METRIC_T_REF="${METRIC_T_REF:-10.0}"

bash start_handcap_remote_inference_metrics.sh \
  --policy-config "${POLICY_CONFIG}" \
  --policy-dir "${POLICY_DIR}" \
  --robot-ip "${FLEXIV_ROBOT_IP}" \
  --local-ip "${FLEXIV_LOCAL_IP}" \
  --prompt "${PROMPT}" \
  --task-name "${TASK_NAME}" \
  --ctrl-freq "${CTRL_FREQ}" \
  --record-hz "${RECORD_HZ}" \
  --steps-per-inference "${STEPS_PER_INFERENCE}" \
  --obs-horizon "${OBS_HORIZON}" \
  --camera-config-path "${CAMERA_CONFIG_PATH}" \
  --action-latency "${ACTION_LATENCY}" \
  --metric-monitor-hz "${METRIC_MONITOR_HZ}" \
  --metric-t-ref "${METRIC_T_REF}" \
  --init-qpos "${FLEXIV_INIT_POSE}" \
  --record-root "${RECORD_ROOT}" \
  --interactive-control \
  --startup-timeout 1800 \
  --use-tactile \
  --tactile-calibration-attempts "${TACTILE_CALIBRATION_ATTEMPTS}" \
  "${TACTILE_CALIBRATION_FLAG[@]}" \
  "${TACTILE_CONFIG_OVERRIDE_ARGS[@]}" \
  "${EXTRA_REMOTE_INFERENCE_ARGS[@]}" \
  "$@"
