import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def unix_time_ms():
    return int(time.time() * 1000)


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


class HTCRemoteController:
    def __init__(self, root_dir, script_path, log_dir, default_cap_id, default_collect_on_start):
        self.root_dir = Path(root_dir).resolve()
        self.script_path = Path(script_path).resolve()
        self.log_dir = Path(log_dir).resolve()
        self.runtime_dir = self.log_dir / "runtime"
        self.default_cap_id = default_cap_id
        self.default_collect_on_start = default_collect_on_start
        self.process = None
        self.current_cap_id = None
        self.current_log_path = None
        self.current_stop_file = None
        self.current_session_id = None
        self.lock = threading.Lock()

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def status(self):
        with self.lock:
            return self._status_locked()

    def start(self, payload, request_receive_ms):
        with self.lock:
            status = self._status_locked()
            if status["running"]:
                self._append_control_log(
                    event="start_ignored_already_running",
                    payload=payload,
                    request_receive_ms=request_receive_ms,
                    extra=status,
                )
                return {
                    **status,
                    "ok": True,
                    "message": f"HTC already running, pid={status.get('pid')}",
                    "started": False,
                }

            cap_id = self._payload_int(payload, "cap_id", self.default_cap_id)
            collect_on_start = parse_bool(payload.get("collect_on_start"), self.default_collect_on_start)
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            script_stem = self.script_path.stem
            script_name = self.script_path.name
            stop_file = self.runtime_dir / f"{script_stem}_{run_id}_{cap_id}.stop"
            log_path = self.log_dir / f"{script_stem}_{run_id}_{cap_id}.log"

            try:
                stop_file.unlink(missing_ok=True)
            except TypeError:
                if stop_file.exists():
                    stop_file.unlink()

            command = [
                sys.executable,
                str(self.script_path),
                "--cap_id",
                str(cap_id),
                "--stop-file",
                str(stop_file),
            ]
            if collect_on_start:
                command.append("--collect-on-start")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            creationflags = 0
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

            try:
                with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
                    log_file.write(f"[remote] starting at {datetime.now(timezone.utc).isoformat()}\n")
                    log_file.write(f"[remote] command: {' '.join(command)}\n")
                process = subprocess.Popen(
                    command,
                    cwd=str(self.root_dir),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                threading.Thread(
                    target=self._relay_process_output,
                    args=(process, log_path, script_name),
                    daemon=True,
                ).start()
            except Exception as exc:
                self._append_control_log(
                    event="start_failed",
                    payload=payload,
                    request_receive_ms=request_receive_ms,
                    extra={"error": repr(exc)},
                )
                return {
                    "ok": False,
                    "message": f"Failed to start {script_name}: {exc}",
                    "running": False,
                    "pid": None,
                    "cap_id": cap_id,
                }

            self.process = process
            self.current_cap_id = cap_id
            self.current_log_path = str(log_path)
            self.current_stop_file = str(stop_file)
            self.current_session_id = payload.get("session_id")

            time.sleep(1.0)
            immediate_return_code = process.poll()
            if immediate_return_code is not None:
                result = {
                    **self._status_locked(),
                    "ok": False,
                    "message": (
                        f"{script_name} exited immediately with code {immediate_return_code}. "
                        f"See log: {log_path}"
                    ),
                    "started": False,
                    "collect_on_start": collect_on_start,
                    "log_tail": self._read_log_tail(log_path),
                }
                self._append_control_log(
                    event="start_failed_exited_immediately",
                    payload=payload,
                    request_receive_ms=request_receive_ms,
                    extra=result,
                )
                return result

            result = {
                **self._status_locked(),
                "ok": True,
                "message": f"Started {script_name}, pid={process.pid}",
                "started": True,
                "collect_on_start": collect_on_start,
            }
            self._append_control_log(
                event="start",
                payload=payload,
                request_receive_ms=request_receive_ms,
                extra=result,
            )
            return result

    def stop(self, payload, request_receive_ms):
        with self.lock:
            status = self._status_locked()
            if not status["running"]:
                self._append_control_log(
                    event="stop_ignored_not_running",
                    payload=payload,
                    request_receive_ms=request_receive_ms,
                    extra=status,
                )
                return {
                    **status,
                    "ok": True,
                    "message": "HTC is not running",
                    "stop_requested": False,
                }

            stop_file = Path(self.current_stop_file) if self.current_stop_file else None
            try:
                if stop_file is not None:
                    stop_file.write_text(str(unix_time_ms()), encoding="utf-8")
            except Exception as exc:
                return {
                    **status,
                    "ok": False,
                    "message": f"Failed to request remote stop: {exc}",
                    "stop_requested": False,
                }

            deadline = time.time() + 3.0
            while time.time() < deadline:
                if self.process is None or self.process.poll() is not None:
                    break
                time.sleep(0.1)

            forced_stop = False
            killed = False
            if self.process is not None and self.process.poll() is None:
                forced_stop = True
                try:
                    self.process.terminate()
                    self.process.wait(timeout=3.0)
                except Exception:
                    if self.process.poll() is None:
                        try:
                            self.process.kill()
                            killed = True
                        except Exception:
                            pass

            result = {
                **self._status_locked(),
                "ok": True,
                "message": self._stop_message(forced_stop=forced_stop, killed=killed),
                "stop_requested": True,
                "forced_stop": forced_stop,
                "killed": killed,
            }
            self._append_control_log(
                event="stop",
                payload=payload,
                request_receive_ms=request_receive_ms,
                extra=result,
            )
            return result

    def _status_locked(self):
        running = False
        return_code = None
        pid = None
        if self.process is not None:
            return_code = self.process.poll()
            running = return_code is None
            pid = self.process.pid

        return {
            "ok": True,
            "message": "Windows USB bridge ready",
            "running": running,
            "pid": pid,
            "return_code": return_code,
            "cap_id": self.current_cap_id,
            "session_id": self.current_session_id,
            "log_path": self.current_log_path,
            "stop_file": self.current_stop_file,
        }

    def _payload_int(self, payload, key, default):
        try:
            return int(payload.get(key, default))
        except (TypeError, ValueError):
            return default

    def _read_log_tail(self, log_path, max_lines=25):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as file:
                lines = file.readlines()
        except Exception as exc:
            return f"<failed to read log tail: {exc}>"
        return "".join(lines[-max_lines:]).strip()

    def _relay_process_output(self, process, log_path, script_name):
        prefix = f"[{script_name}] "
        try:
            with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
                stream = process.stdout
                if stream is None:
                    return
                for line in stream:
                    log_file.write(line)
                    print(prefix + line.rstrip("\r\n"), flush=True)
        except Exception as exc:
            print(f"[remote] failed to relay {script_name} output: {exc}", flush=True)

    def _stop_message(self, forced_stop, killed):
        if killed:
            return f"Stop requested for {self.script_path.name}; process had to be killed"
        if forced_stop:
            return f"Stop requested for {self.script_path.name}; process was terminated after timeout"
        return f"Stop requested for {self.script_path.name}"

    def _append_control_log(self, event, payload, request_receive_ms, extra):
        record = {
            "event": event,
            "server_receive_unix_time_ms": request_receive_ms,
            "server_log_unix_time_ms": unix_time_ms(),
            "payload": payload,
            "extra": extra,
        }
        log_path = self.log_dir / "remote_control_events.jsonl"
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")
