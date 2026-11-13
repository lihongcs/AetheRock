#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./start_handcap_remote_inference_metrics.sh \
    --policy-config <handcap_config_name> \
    --policy-dir <checkpoint_dir> \
    --robot-ip <flexiv_ip> \
    --prompt "<task_prompt>" \
    [--local-ip <local_ip>] \
    [--server-port 8000] \
    [--task-name handcap_flexiv_mvs_metrics] \
    [--ctrl-freq 20] \
    [--record-hz 20] \
    [--steps-per-inference 4] \
    [--obs-horizon 2] \
    [--camera-config-path <path>] \
    [--use-tactile | --no-use-tactile] \
    [--left-video-index <idx>] \
    [--right-video-index <idx>] \
    [--tactile-width 224] \
    [--tactile-height 224] \
    [--tactile-capture-width 640] \
    [--tactile-capture-height 480] \
    [--tactile-calibration-attempts 120] \
    [--tactile-calibration | --no-tactile-calibration] \
    [--force-predict | --no-force-predict] \
    [--force-guide | --no-force-guide] \
    [--infer-force-control | --no-infer-force-control] \
    [--action-latency 0.0] \
    [--metric-monitor-hz 50] \
    [--metric-t-ref 10.0] \
    [--init-qpos "<7 floats, comma list, or bracket list>"] \
    [--record-root <dir>] \
    [--server-default-prompt "<prompt>"] \
    [--interactive-control] \
    [--control-file <path>] \
    [--startup-timeout 120] \
    [--log-dir <dir>] \
    [--dry-run]

Required:
  --policy-config       Handcap config name registered in openpi training config.
  --policy-dir          Checkpoint directory to serve.
  --robot-ip            Flexiv robot IP.
  --prompt              Task prompt sent to the policy server.
EOF
}

require_value() {
  local flag="$1"
  local value="${2-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    echo "Missing value for $flag" >&2
    usage
    exit 1
  fi
}

join_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    out+="$(printf '%q' "$arg") "
  done
  printf '%s\n' "${out% }"
}

is_port_open() {
  local host="$1"
  local port="$2"

  python3 - "$host" "$port" <<'PY' >/dev/null 2>&1
import base64
import os
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

key = base64.b64encode(os.urandom(16)).decode("ascii")
request = (
    f"GET / HTTP/1.1\r\n"
    f"Host: {host}:{port}\r\n"
    "Upgrade: websocket\r\n"
    "Connection: Upgrade\r\n"
    f"Sec-WebSocket-Key: {key}\r\n"
    "Sec-WebSocket-Version: 13\r\n"
    "\r\n"
)

with socket.create_connection((host, port), timeout=1) as sock:
    sock.settimeout(1)
    sock.sendall(request.encode("ascii"))
    response = sock.recv(1024)

status_line = response.split(b"\r\n", 1)[0]
if b" 101 " not in status_line:
    raise SystemExit(1)
PY
}

is_process_running() {
  local pid="$1"
  local state=""

  if ! kill -0 "$pid" >/dev/null 2>&1; then
    return 1
  fi

  state="$(ps -p "$pid" -o stat= 2>/dev/null || true)"
  [[ -n "$state" && "$state" != Z* ]]
}

log_process_snapshot() {
  local reason="${1:-snapshot}"
  local log_file="${PROCESS_MONITOR_LOG:-}"
  local ps_args=("-p" "$$")

  if [[ -z "$log_file" ]]; then
    return
  fi

  if [[ -n "${SERVER_PID:-}" ]]; then
    ps_args+=("-p" "$SERVER_PID")
  fi
  if [[ -n "${CLIENT_PID:-}" ]]; then
    ps_args+=("-p" "$CLIENT_PID")
  fi

  {
    echo "===== $(date '+%F %T') :: ${reason} ====="
    echo "launcher_pid=$$ launcher_ppid=$PPID server_pid=${SERVER_PID:-} client_pid=${CLIENT_PID:-}"
    echo "--- process table ---"
    ps -o pid,ppid,pgid,sid,stat,etime,%cpu,%mem,rss,vsz,cmd \
      "${ps_args[@]}" \
      2>/dev/null || true
    echo "--- process tree ---"
    if command -v pstree >/dev/null 2>&1; then
      pstree -aps "$$" 2>/dev/null || true
      if [[ -n "${SERVER_PID:-}" ]]; then
        pstree -aps "$SERVER_PID" 2>/dev/null || true
      fi
      if [[ -n "${CLIENT_PID:-}" ]]; then
        pstree -aps "$CLIENT_PID" 2>/dev/null || true
      fi
    fi
    echo "--- matching inference processes ---"
    ps -ef | grep -E "run_inference_pi05|start_handcap_remote_inference|serve_policy|handcap_flexiv_remote_metrics" | grep -v grep || true
    if command -v pstree >/dev/null 2>&1; then
      :
    else
      echo "pstree not found"
    fi
    echo "--- system memory ---"
    free -h 2>/dev/null || true
    echo "--- top memory processes ---"
    ps -eo pid,ppid,stat,%mem,rss,vsz,cmd --sort=-rss 2>/dev/null | head -20 || true
    echo "--- gpu summary ---"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true
      echo "--- gpu processes ---"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
    else
      echo "nvidia-smi not found"
    fi
    echo
  } >> "$log_file" 2>&1
}

start_process_monitor() {
  if [[ "${INFERENCE_DEBUG_MONITOR:-1}" != "1" ]]; then
    return
  fi

  PROCESS_MONITOR_LOG="${PROCESS_MONITOR_LOG:-${LOG_DIR}/process_monitor.log}"
  export PROCESS_MONITOR_LOG
  echo "Process monitor log: $PROCESS_MONITOR_LOG"

  (
    while true; do
      log_process_snapshot "periodic"
      sleep "${INFERENCE_DEBUG_MONITOR_INTERVAL:-2}"
    done
  ) &
  PROCESS_MONITOR_PID=$!
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout="$3"
  local pid="${4-}"
  local log_file="${5-}"
  local start_time=$SECONDS
  local deadline=$((SECONDS + timeout))
  local next_status=$((SECONDS + 20))

  while (( SECONDS < deadline )); do
    if is_port_open "$host" "$port"; then
      return 0
    fi

    if [[ -n "$pid" ]] && ! is_process_running "$pid"; then
      wait "$pid" || true
      return 2
    fi

    if (( SECONDS >= next_status )); then
      echo "Still waiting for server on ${host}:${port} ... ($((SECONDS - start_time))s elapsed)"
      if [[ -n "$log_file" && -s "$log_file" ]]; then
        echo "==== server.log (tail) ===="
        tail -n 20 "$log_file" || true
        echo "==========================="
      fi
      next_status=$((SECONDS + 20))
    fi

    sleep 1
  done
  return 1
}

POLICY_CONFIG=""
POLICY_DIR=""
ROBOT_IP="${FLEXIV_ROBOT_IP:-}"
LOCAL_IP="${FLEXIV_LOCAL_IP:-}"
PROMPT=""
SERVER_PORT="8000"
TASK_NAME="handcap_flexiv_mvs_metrics"
CTRL_FREQ="20"
RECORD_HZ="20"
STEPS_PER_INFERENCE="4"
OBS_HORIZON="2"
CAMERA_CONFIG_PATH=""
USE_TACTILE="true"
FORCE_PREDICT="false"
FORCE_GUIDE="false"
ACTION_LATENCY="0.0"
INIT_QPOS=""
RECORD_ROOT=""
SERVER_DEFAULT_PROMPT=""
INTERACTIVE_CONTROL="false"
CONTROL_FILE=""
STARTUP_TIMEOUT="600"
LOG_DIR=""
DRY_RUN="false"
LEFT_VIDEO_INDEX=""
RIGHT_VIDEO_INDEX=""
TACTILE_WIDTH="224"
TACTILE_HEIGHT="224"
TACTILE_CAPTURE_WIDTH=""
TACTILE_CAPTURE_HEIGHT=""
TACTILE_CALIBRATION="true"
TACTILE_CALIBRATION_ATTEMPTS="120"
TACTILE_CONFIG_OVERRIDE="false"
INFER_FORCE_CONTROL="false"
METRIC_MONITOR_HZ="50"
METRIC_T_REF="10.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-config)
      require_value "$1" "${2-}"
      POLICY_CONFIG="$2"
      shift 2
      ;;
    --policy-dir)
      require_value "$1" "${2-}"
      POLICY_DIR="$2"
      shift 2
      ;;
    --robot-ip)
      require_value "$1" "${2-}"
      ROBOT_IP="$2"
      shift 2
      ;;
    --local-ip)
      require_value "$1" "${2-}"
      LOCAL_IP="$2"
      shift 2
      ;;
    --prompt)
      require_value "$1" "${2-}"
      PROMPT="$2"
      shift 2
      ;;
    --server-port)
      require_value "$1" "${2-}"
      SERVER_PORT="$2"
      shift 2
      ;;
    --task-name)
      require_value "$1" "${2-}"
      TASK_NAME="$2"
      shift 2
      ;;
    --ctrl-freq)
      require_value "$1" "${2-}"
      CTRL_FREQ="$2"
      shift 2
      ;;
    --record-hz)
      require_value "$1" "${2-}"
      RECORD_HZ="$2"
      shift 2
      ;;
    --steps-per-inference)
      require_value "$1" "${2-}"
      STEPS_PER_INFERENCE="$2"
      shift 2
      ;;
    --obs-horizon)
      require_value "$1" "${2-}"
      OBS_HORIZON="$2"
      shift 2
      ;;
    --camera-config-path)
      require_value "$1" "${2-}"
      CAMERA_CONFIG_PATH="$2"
      shift 2
      ;;
    --use-tactile)
      USE_TACTILE="true"
      shift
      ;;
    --no-use-tactile)
      USE_TACTILE="false"
      shift
      ;;
    --left-video-index)
      require_value "$1" "${2-}"
      LEFT_VIDEO_INDEX="$2"
      TACTILE_CONFIG_OVERRIDE="true"
      shift 2
      ;;
    --right-video-index)
      require_value "$1" "${2-}"
      RIGHT_VIDEO_INDEX="$2"
      TACTILE_CONFIG_OVERRIDE="true"
      shift 2
      ;;
    --tactile-width)
      require_value "$1" "${2-}"
      TACTILE_WIDTH="$2"
      TACTILE_CONFIG_OVERRIDE="true"
      shift 2
      ;;
    --tactile-height)
      require_value "$1" "${2-}"
      TACTILE_HEIGHT="$2"
      TACTILE_CONFIG_OVERRIDE="true"
      shift 2
      ;;
    --tactile-capture-width)
      require_value "$1" "${2-}"
      TACTILE_CAPTURE_WIDTH="$2"
      TACTILE_CONFIG_OVERRIDE="true"
      shift 2
      ;;
    --tactile-capture-height)
      require_value "$1" "${2-}"
      TACTILE_CAPTURE_HEIGHT="$2"
      TACTILE_CONFIG_OVERRIDE="true"
      shift 2
      ;;
    --tactile-calibration)
      TACTILE_CALIBRATION="true"
      shift
      ;;
    --no-tactile-calibration)
      TACTILE_CALIBRATION="false"
      shift
      ;;
    --tactile-calibration-attempts)
      require_value "$1" "${2-}"
      TACTILE_CALIBRATION_ATTEMPTS="$2"
      shift 2
      ;;
    --force-predict)
      FORCE_PREDICT="true"
      shift
      ;;
    --no-force-predict)
      FORCE_PREDICT="false"
      shift
      ;;
    --force-guide)
      FORCE_GUIDE="true"
      shift
      ;;
    --no-force-guide)
      FORCE_GUIDE="false"
      shift
      ;;
    --infer-force-control)
      INFER_FORCE_CONTROL="true"
      shift
      ;;
    --no-infer-force-control)
      INFER_FORCE_CONTROL="false"
      shift
      ;;
    --action-latency)
      require_value "$1" "${2-}"
      ACTION_LATENCY="$2"
      shift 2
      ;;
    --metric-monitor-hz)
      require_value "$1" "${2-}"
      METRIC_MONITOR_HZ="$2"
      shift 2
      ;;
    --metric-t-ref)
      require_value "$1" "${2-}"
      METRIC_T_REF="$2"
      shift 2
      ;;
    --init-qpos)
      require_value "$1" "${2-}"
      INIT_QPOS="$2"
      shift 2
      ;;
    --record-root)
      require_value "$1" "${2-}"
      RECORD_ROOT="$2"
      shift 2
      ;;
    --server-default-prompt)
      require_value "$1" "${2-}"
      SERVER_DEFAULT_PROMPT="$2"
      shift 2
      ;;
    --interactive-control)
      INTERACTIVE_CONTROL="true"
      shift
      ;;
    --control-file)
      require_value "$1" "${2-}"
      CONTROL_FILE="$2"
      shift 2
      ;;
    --startup-timeout)
      require_value "$1" "${2-}"
      STARTUP_TIMEOUT="$2"
      shift 2
      ;;
    --log-dir)
      require_value "$1" "${2-}"
      LOG_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$POLICY_CONFIG" || -z "$POLICY_DIR" || -z "$ROBOT_IP" || -z "$PROMPT" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ "$POLICY_DIR" == "~/"* ]]; then
  POLICY_DIR="$HOME/${POLICY_DIR#~/}"
fi

if [[ "$POLICY_DIR" != gs://* && "$POLICY_DIR" != s3://* && ! -d "$POLICY_DIR" ]]; then
  echo "Cannot find policy checkpoint directory: $POLICY_DIR" >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/scripts/serve_policy.py" ]]; then
  echo "Cannot find $REPO_ROOT/scripts/serve_policy.py" >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/scripts/handcap_flexiv_remote_metrics.py" ]]; then
  echo "Cannot find $REPO_ROOT/scripts/handcap_flexiv_remote_metrics.py" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Activate the ForceVT Python environment or start this launcher with uv run bash." >&2
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/packages/openpi-client/src:${PYTHONPATH:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$REPO_ROOT/logs/handcap_remote_inference/$TIMESTAMP"
fi
mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/server.log"
CLIENT_LOG="$LOG_DIR/client.log"
PID_FILE="$LOG_DIR/pids.env"
CMD_FILE="$LOG_DIR/run_command.txt"

if [[ "$INTERACTIVE_CONTROL" == "true" && -z "$CONTROL_FILE" ]]; then
  CONTROL_FILE="$LOG_DIR/runtime_control.commands"
fi
CONTROL_EVENT_FILE=""
if [[ -n "$CONTROL_FILE" ]]; then
  mkdir -p "$(dirname "$CONTROL_FILE")"
  : > "$CONTROL_FILE"
  CONTROL_EVENT_FILE="${CONTROL_FILE}.events"
  : > "$CONTROL_EVENT_FILE"
fi

if [[ "$TACTILE_CONFIG_OVERRIDE" == "true" ]]; then
  CUSTOM_CONFIG_PATH="$LOG_DIR/custom_camera_config.json"
  python -c "
import json
import sys
import os

configured_path = '$CAMERA_CONFIG_PATH'
default_path = os.path.abspath(configured_path if configured_path else 'src/perception/configs/camera/handcap_realsense_tactile.json')
if not os.path.exists(default_path):
    print(f'Warning: Could not find default camera config at {default_path}')
    sys.exit(0)

with open(default_path, 'r') as f:
    config = json.load(f)

def parse_index(value):
    return int(value) if str(value).isdigit() else value

def ensure_tactile_camera(name, index_value):
    if name not in config and not index_value:
        return
    entry = config.setdefault(name, {})
    entry.setdefault('type', 'tactile_opencv')
    if index_value:
        entry['camera_index'] = parse_index(index_value)
    if 'fps' not in entry:
        entry['fps'] = 20
    entry['resolution'] = [int('$TACTILE_WIDTH'), int('$TACTILE_HEIGHT')]
    entry.setdefault('capture_resolution', [640, 480])

ensure_tactile_camera('left_tactile', '$LEFT_VIDEO_INDEX')
ensure_tactile_camera('right_tactile', '$RIGHT_VIDEO_INDEX')

capture_width = '$TACTILE_CAPTURE_WIDTH'
capture_height = '$TACTILE_CAPTURE_HEIGHT'
if capture_width or capture_height:
    if not (capture_width and capture_height):
        raise ValueError('Both tactile capture width and height must be set together')
    capture_resolution = [int(capture_width), int(capture_height)]
    for name in ('left_tactile', 'right_tactile'):
        if name in config:
            config[name]['capture_resolution'] = capture_resolution

with open('$CUSTOM_CONFIG_PATH', 'w') as f:
    json.dump(config, f, indent=4)
"
  if [[ -f "$CUSTOM_CONFIG_PATH" ]]; then
    CAMERA_CONFIG_PATH="$CUSTOM_CONFIG_PATH"
  fi
fi

if [[ "$USE_TACTILE" == "true" ]]; then
  TACTILE_PREFLIGHT_CONFIG="$CAMERA_CONFIG_PATH"
  if [[ -z "$TACTILE_PREFLIGHT_CONFIG" ]]; then
    TACTILE_PREFLIGHT_CONFIG="src/perception/configs/camera/handcap_realsense_tactile.json"
  fi
  python - "$TACTILE_PREFLIGHT_CONFIG" <<'PY'
import json
import os
import sys
import time

import cv2

config_path = sys.argv[1]
if not os.path.isabs(config_path):
    config_path = os.path.abspath(config_path)
if not os.path.exists(config_path):
    raise SystemExit(f"Tactile preflight failed: camera config not found: {config_path}")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

missing = [name for name in ("left_tactile", "right_tactile") if name not in config]
if missing:
    raise SystemExit(
        "Tactile preflight failed: use_tactile=True requires "
        f"{', '.join(missing)} in {config_path}"
    )


def device_from_entry(entry):
    if entry.get("camera_v4l_path"):
        return entry["camera_v4l_path"]
    index = entry.get("camera_index")
    if index is None:
        raise ValueError("missing camera_index or camera_v4l_path")
    if isinstance(index, str) and index.startswith("/"):
        return index
    return f"/dev/video{index}"


def check_camera(name, entry):
    if entry.get("type") not in ("tactile_opencv", "opencv"):
        raise ValueError(f"unsupported tactile camera type {entry.get('type')!r}")
    device = device_from_entry(entry)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FPS, float(entry.get("fps", 20)))
        capture_resolution = entry.get("capture_resolution") or [640, 480]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(capture_resolution[0]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(capture_resolution[1]))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        last_shape = None
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                last_shape = frame.shape
                print(f"Tactile preflight OK: {name} {device} shape={last_shape}")
                return
            time.sleep(0.05)
        raise RuntimeError(f"opened {device}, but could not read a valid frame; last_shape={last_shape}")
    finally:
        cap.release()


for side in ("left_tactile", "right_tactile"):
    try:
        check_camera(side, config[side])
    except Exception as exc:
        raise SystemExit(f"Tactile preflight failed: {side}: {exc}") from exc
PY
else
  echo "use_tactile=false: skipping tactile camera preflight."
fi

SERVER_CMD=(
  python
  scripts/serve_policy.py
  --port="$SERVER_PORT"
  policy:checkpoint
  --policy.config="$POLICY_CONFIG"
  --policy.dir="$POLICY_DIR"
)

if [[ -n "$SERVER_DEFAULT_PROMPT" ]]; then
  SERVER_CMD=("${SERVER_CMD[@]:0:3}" --default-prompt="$SERVER_DEFAULT_PROMPT" "${SERVER_CMD[@]:3}")
fi

CLIENT_CMD=(
  python
  scripts/handcap_flexiv_remote_metrics.py
  --host 127.0.0.1
  --port "$SERVER_PORT"
  --robot-ip "$ROBOT_IP"
  --local-ip "$LOCAL_IP"
  --prompt "$PROMPT"
  --task-name "$TASK_NAME"
  --ctrl-freq "$CTRL_FREQ"
  --record-hz "$RECORD_HZ"
  --steps-per-inference "$STEPS_PER_INFERENCE"
  --obs-horizon "$OBS_HORIZON"
  --action-latency "$ACTION_LATENCY"
  --metric-monitor-hz "$METRIC_MONITOR_HZ"
  --metric-t-ref "$METRIC_T_REF"
  --tactile-calibration-attempts "$TACTILE_CALIBRATION_ATTEMPTS"
)

if [[ -n "$CAMERA_CONFIG_PATH" ]]; then
  CLIENT_CMD+=(--camera-config-path "$CAMERA_CONFIG_PATH")
fi

if [[ "$USE_TACTILE" == "true" ]]; then
  CLIENT_CMD+=(--use-tactile)
else
  CLIENT_CMD+=(--no-use-tactile)
fi

if [[ "$TACTILE_CALIBRATION" != "true" ]]; then
  CLIENT_CMD+=(--no-tactile-calibration)
fi

if [[ "$FORCE_PREDICT" == "true" ]]; then
  CLIENT_CMD+=(--force-predict)
fi

if [[ "$FORCE_GUIDE" == "true" ]]; then
  CLIENT_CMD+=(--force-guide)
fi

if [[ "$INFER_FORCE_CONTROL" == "true" ]]; then
  CLIENT_CMD+=(--infer-force-control)
fi

if [[ -n "$INIT_QPOS" ]]; then
  INIT_QPOS_NORMALIZED="${INIT_QPOS//,/ }"
  INIT_QPOS_NORMALIZED="${INIT_QPOS_NORMALIZED//[/ }"
  INIT_QPOS_NORMALIZED="${INIT_QPOS_NORMALIZED//]/ }"
  read -r -a INIT_QPOS_ARGS <<< "$INIT_QPOS_NORMALIZED"
  CLIENT_CMD+=(--init-qpos "${INIT_QPOS_ARGS[@]}")
fi

if [[ -n "$RECORD_ROOT" ]]; then
  CLIENT_CMD+=(--record-root "$RECORD_ROOT")
fi

if [[ -n "$CONTROL_FILE" ]]; then
  CLIENT_CMD+=(--control-file "$CONTROL_FILE")
fi

if [[ "$DRY_RUN" == "true" ]]; then
  CLIENT_CMD+=(--dry-run)
fi

CLIENT_LD_PRELOAD="${LD_PRELOAD:-}"
SYSTEM_LIBSTDCXX="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if [[ -f "$SYSTEM_LIBSTDCXX" ]] && grep -a -q "GLIBCXX_3.4.30" "$SYSTEM_LIBSTDCXX"; then
  CLIENT_LD_PRELOAD="$SYSTEM_LIBSTDCXX${CLIENT_LD_PRELOAD:+:$CLIENT_LD_PRELOAD}"
fi

{
  echo "# Generated at $(date --iso-8601=seconds)"
  echo "Debug monitor: INFERENCE_DEBUG_MONITOR=${INFERENCE_DEBUG_MONITOR:-1}, interval=${INFERENCE_DEBUG_MONITOR_INTERVAL:-2}s"
  echo "Python env: PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1"
  echo
  echo "Resource limits:"
  ulimit -a
  echo
  echo "Selected environment:"
  env | grep -E '^(CUDA_VISIBLE_DEVICES|JAX_|XLA_|TF_|LD_PRELOAD|FLEXIV_|RECORD_|POLICY_|INFERENCE_DEBUG_)=' | sort || true
  echo
  echo "Server command:"
  join_cmd "${SERVER_CMD[@]}"
  echo
  echo "Client command:"
  if [[ -n "$CLIENT_LD_PRELOAD" ]]; then
    echo "LD_PRELOAD=$(printf '%q' "$CLIENT_LD_PRELOAD") $(join_cmd "${CLIENT_CMD[@]}")"
  else
    join_cmd "${CLIENT_CMD[@]}"
  fi
} > "$CMD_FILE"

SERVER_PID=""
CLIENT_PID=""
CLIENT_LOG_TAIL_PID=""
PROCESS_MONITOR_PID=""

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  if [[ -n "${PROCESS_MONITOR_PID:-}" ]] && kill -0 "$PROCESS_MONITOR_PID" >/dev/null 2>&1; then
    log_process_snapshot "cleanup exit_code=${exit_code}"
    kill "$PROCESS_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$PROCESS_MONITOR_PID" 2>/dev/null || true
  fi

  if [[ -n "${CLIENT_LOG_TAIL_PID:-}" ]] && kill -0 "$CLIENT_LOG_TAIL_PID" >/dev/null 2>&1; then
    kill "$CLIENT_LOG_TAIL_PID" >/dev/null 2>&1 || true
    wait "$CLIENT_LOG_TAIL_PID" 2>/dev/null || true
  fi

  if [[ -n "${CLIENT_PID:-}" ]] && kill -0 "$CLIENT_PID" >/dev/null 2>&1; then
    echo "Stopping client process $CLIENT_PID and waiting for recorder flush..."
    kill -INT "$CLIENT_PID" >/dev/null 2>&1 || true
    for _ in {1..30}; do
      if ! is_process_running "$CLIENT_PID"; then
        break
      fi
      sleep 0.5
    done
    if is_process_running "$CLIENT_PID"; then
      echo "Client did not exit after flush wait; sending TERM."
      kill -TERM "$CLIENT_PID" >/dev/null 2>&1 || true
    fi
    wait "$CLIENT_PID" 2>/dev/null || true
  fi

  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi

  exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "Starting OpenPI policy server..."
echo "Server log: $SERVER_LOG"
echo "Server command: $(join_cmd "${SERVER_CMD[@]}")"
if is_port_open "127.0.0.1" "$SERVER_PORT"; then
  echo "Port 127.0.0.1:$SERVER_PORT is already open. Stop the old server or use --server-port." >&2
  exit 1
fi
PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 "${SERVER_CMD[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for server on 127.0.0.1:$SERVER_PORT ..."
WAIT_STATUS=0
wait_for_port "127.0.0.1" "$SERVER_PORT" "$STARTUP_TIMEOUT" "$SERVER_PID" "$SERVER_LOG" || WAIT_STATUS=$?
if [[ "$WAIT_STATUS" != "0" ]]; then
  if [[ "$WAIT_STATUS" == "2" ]]; then
    echo "Server process exited before opening 127.0.0.1:$SERVER_PORT." >&2
  else
    echo "Server did not become ready within ${STARTUP_TIMEOUT}s." >&2
  fi
  if [[ -f "$SERVER_LOG" ]]; then
    echo "==== server.log (tail) ====" >&2
    tail -n 50 "$SERVER_LOG" >&2 || true
  fi
  exit 1
fi
echo "Policy server is ready; pretrained weights are loaded."

echo "Starting Flexiv real-robot client..."
if [[ -n "$CLIENT_LD_PRELOAD" ]]; then
  echo "Client LD_PRELOAD: $CLIENT_LD_PRELOAD"
  LD_PRELOAD="$CLIENT_LD_PRELOAD" PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 "${CLIENT_CMD[@]}" >"$CLIENT_LOG" 2>&1 &
else
  PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 "${CLIENT_CMD[@]}" >"$CLIENT_LOG" 2>&1 &
fi
CLIENT_PID=$!

{
  echo "SERVER_PID=$SERVER_PID"
  echo "CLIENT_PID=$CLIENT_PID"
} > "$PID_FILE"

start_process_monitor

cat <<EOF
Remote inference started.

Logs:
  Server: $SERVER_LOG
  Client: $CLIENT_LOG
  Process monitor: ${PROCESS_MONITOR_LOG:-disabled}

PID file:
  $PID_FILE
EOF
if [[ "$INTERACTIVE_CONTROL" == "true" ]]; then
  cat <<EOF

Controls:
  SPACE  start current episode; press SPACE again to stop it and choose save/delete
  r      reset robot to init pose while idle or at the next safe point
  Ctrl+C exit the whole loaded inference session
EOF
else
  echo "Press Ctrl+C to stop both processes."
fi
if [[ -n "$CONTROL_FILE" ]]; then
  echo "Control file: $CONTROL_FILE"
fi

echo "==== live client.log ===="
tail -n 0 -F "$CLIENT_LOG" &
CLIENT_LOG_TAIL_PID=$!

CONTROL_EVENT_OFFSET=0
EPISODE_RUNNING=0
CONTROL_EVENT_DELIM=$'\t'

wait_for_episode_finished() {
  local timeout="${1:-300}"
  local deadline=$((SECONDS + timeout))
  local line=""
  local event_name=""
  local event_payload=""
  local current_size=0

  if [[ -z "$CONTROL_EVENT_FILE" ]]; then
    return 1
  fi

  while (( SECONDS < deadline )); do
    if ! is_process_running "$CLIENT_PID"; then
      return 2
    fi

    if [[ -f "$CONTROL_EVENT_FILE" ]]; then
      current_size=$(wc -c < "$CONTROL_EVENT_FILE" | tr -d ' ')
      if (( current_size > CONTROL_EVENT_OFFSET )); then
        while IFS="$CONTROL_EVENT_DELIM" read -r event_name event_payload; do
          if [[ "$event_name" == "episode_finished" && -n "$event_payload" ]]; then
            EPISODE_OUTPUT_DIR="$event_payload"
            CONTROL_EVENT_OFFSET=$current_size
            return 0
          fi
        done < <(tail -c +"$((CONTROL_EVENT_OFFSET + 1))" "$CONTROL_EVENT_FILE")
        CONTROL_EVENT_OFFSET=$current_size
      fi
    fi
    sleep 0.2
  done

  return 1
}

prompt_keep_episode() {
  local episode_dir="$1"
  local keep_data=""

  if [[ -z "$episode_dir" || ! -d "$episode_dir" ]]; then
    echo "未发现本轮真机推理数据目录，跳过保存确认。"
    printf 'discard\n' >> "$CONTROL_FILE"
    return
  fi

  echo ""
  read -p "❓ 本轮真机推理数据是否需要保留? 输入 y 保留，直接回车或其他键删除 [y/N]: " keep_data </dev/tty
  case "$keep_data" in
    y|Y|yes|Yes )
      echo "✅ 已保留本轮数据: $episode_dir"
      printf 'keep\n' >> "$CONTROL_FILE"
      
      local success_ratio=""
      read -p "❓ 请输入成功率 (success ratio): " success_ratio </dev/tty
      
      local episode_name=$(basename "$episode_dir")
      local json_path="$(dirname "$episode_dir")/success_ratios.json"
      python -c "
import json, os
path = '$json_path'
data = {}
if os.path.exists(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except:
        pass
episode_num = '$episode_name'.replace('episode_', '')
data[episode_num] = '$success_ratio'
with open(path, 'w') as f:
    json.dump(data, f, indent=4)
"
      echo "✅ 已保存成功率到: $json_path"
      ;;
    * )
      rm -rf "$episode_dir"
      echo "🗑️  已删除本轮数据: $episode_dir"
      printf 'discard\n' >> "$CONTROL_FILE"
      ;;
  esac
}

EXITED_PROCESS=""
EXIT_STATUS=0
while true; do
  if ! is_process_running "$SERVER_PID"; then
    log_process_snapshot "detected server exit before wait"
    if wait "$SERVER_PID"; then
      EXIT_STATUS=0
    else
      EXIT_STATUS=$?
    fi
    log_process_snapshot "server wait returned status ${EXIT_STATUS}"
    EXITED_PROCESS="server"
    break
  fi

  if ! is_process_running "$CLIENT_PID"; then
    log_process_snapshot "detected client exit before wait"
    if wait "$CLIENT_PID"; then
      EXIT_STATUS=0
    else
      EXIT_STATUS=$?
    fi
    log_process_snapshot "client wait returned status ${EXIT_STATUS}"
    EXITED_PROCESS="client"
    break
  fi

  if [[ "$INTERACTIVE_CONTROL" == "true" && -r /dev/tty ]]; then
    key=""
    if IFS= read -r -s -n 1 -t 1 key </dev/tty; then
      case "$key" in
        " ")
          if [[ "$EPISODE_RUNNING" == "0" ]]; then
            printf 'start\n' >> "$CONTROL_FILE"
            EPISODE_RUNNING=1
            echo "Sent start command: recording and inference will begin when the client reaches a safe point."
          else
            printf 'stop\n' >> "$CONTROL_FILE"
            echo "Sent stop command: finishing the current episode and flushing video..."
            EPISODE_OUTPUT_DIR=""
            if wait_for_episode_finished 600; then
              EPISODE_RUNNING=0
              prompt_keep_episode "$EPISODE_OUTPUT_DIR"
              echo "Ready for the next episode: press r to reset or SPACE to start."
            else
              wait_status=$?
              EPISODE_RUNNING=0
              if [[ "$wait_status" == "2" ]]; then
                echo "Client exited before reporting episode completion." >&2
              else
                echo "Timed out waiting for episode completion. Check client.log before deleting data." >&2
              fi
            fi
          fi
          ;;
        r|R)
          printf 'reset\n' >> "$CONTROL_FILE"
          echo "Sent reset command: robot will move to init pose when the client reaches a safe point."
          ;;
      esac
    fi
  else
    sleep 1
  fi
done

if [[ -n "${CLIENT_LOG_TAIL_PID:-}" ]] && kill -0 "$CLIENT_LOG_TAIL_PID" >/dev/null 2>&1; then
  kill "$CLIENT_LOG_TAIL_PID" >/dev/null 2>&1 || true
  wait "$CLIENT_LOG_TAIL_PID" 2>/dev/null || true
fi

if [[ "$EXIT_STATUS" == "0" ]]; then
  echo "$EXITED_PROCESS process exited normally. Stopping the remaining process..."
else
  echo "$EXITED_PROCESS process exited with status $EXIT_STATUS. Stopping the remaining process..." >&2
fi

if [[ -f "$SERVER_LOG" ]]; then
  echo "==== server.log (tail) ===="
  tail -n 80 "$SERVER_LOG" || true
fi

if [[ -f "$CLIENT_LOG" ]]; then
  echo "==== client.log (tail) ===="
  tail -n 80 "$CLIENT_LOG" || true
fi
