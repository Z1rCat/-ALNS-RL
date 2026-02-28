import base64
import csv
import concurrent.futures
import datetime
import json
import os
import shutil
import smtplib
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[2]
CODES_DIR = ROOT_DIR / "codes"
LOG_ROOT = CODES_DIR / "logs"
MASTER_OUTPUT_FILES = ("rl_trace.csv", "rl_training.csv", "rl_summary.csv", "console_output.txt")
BASELINE_OUTPUT_FILES = ("baseline_wait.csv", "baseline_reroute.csv", "baseline_random.csv")
DONE_MARKER_FILE = "DONE.json"
LOCK_FILE = "run.lock"
HEARTBEAT_FILE = "heartbeat.json"


_ANSI_ENABLED = bool(sys.stdout and sys.stdout.isatty())


def _enable_ansi_on_windows() -> None:
    global _ANSI_ENABLED
    if os.name != "nt" or not _ANSI_ENABLED:
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        _ANSI_ENABLED = False


def _c(text: str, color: str = "", bold: bool = False) -> str:
    if not _ANSI_ENABLED:
        return text
    codes: List[str] = []
    if bold:
        codes.append("1")
    if color:
        palette = {
            "red": "31",
            "green": "32",
            "yellow": "33",
            "blue": "34",
            "magenta": "35",
            "cyan": "36",
            "white": "37",
        }
        value = palette.get(color.lower())
        if value:
            codes.append(value)
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


_enable_ansi_on_windows()


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _file_has_content(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= int(min_bytes)
    except Exception:
        return False


def _csv_has_data_row(path: Path, min_rows: int = 1) -> bool:
    if not _file_has_content(path, min_bytes=8):
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            # skip header
            try:
                next(reader)
            except StopIteration:
                return False
            count = 0
            for row in reader:
                if not row:
                    continue
                if all((str(cell).strip() == "") for cell in row):
                    continue
                count += 1
                if count >= int(min_rows):
                    return True
        return False
    except Exception:
        return False


def _csv_has_phase_row(path: Path, phase: str, min_rows: int = 1) -> bool:
    if not _file_has_content(path, min_bytes=8):
        return False
    target = str(phase).strip().lower()
    if not target:
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return False
            field_map = {str(name).strip().lower(): name for name in reader.fieldnames if name}
            phase_key = field_map.get("phase")
            if not phase_key:
                return False
            count = 0
            for row in reader:
                value = str(row.get(phase_key, "")).strip().lower()
                if value != target:
                    continue
                count += 1
                if count >= int(min_rows):
                    return True
        return False
    except Exception:
        return False


def _dir_has_content(path: Path) -> bool:
    try:
        return path.exists() and path.is_dir() and any(path.iterdir())
    except Exception:
        return False


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _safe_rmtree(path: Path) -> None:
    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
    except Exception:
        pass


def _safe_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _done_marker_path(run_dir: Path) -> Path:
    return run_dir / DONE_MARKER_FILE


def _write_done_marker(run_dir: Path, payload: Optional[Dict[str, Any]] = None) -> None:
    data = dict(payload or {})
    data.setdefault("status", "done")
    data.setdefault("ts", time.time())
    try:
        _atomic_write_json(_done_marker_path(run_dir), data)
    except Exception:
        pass


def _clear_done_marker(run_dir: Path) -> None:
    _safe_unlink(_done_marker_path(run_dir))


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _format_task_key(task: Tuple[str, int, str, int]) -> str:
    dist_name, request_number, algorithm, seed = task
    return f"{dist_name}|R{request_number}|{algorithm}|S{seed}"


class ResourceMonitor:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Dict[str, Any]] = []
        self._notes: List[str] = []

        self._psutil = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
        except Exception:
            self._psutil = None

        self._has_proc = Path("/proc/stat").exists() and Path("/proc/meminfo").exists()
        self._prev_cpu_total: Optional[int] = None
        self._prev_cpu_idle: Optional[int] = None

        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._psutil is None and not self._has_proc and not self._has_nvidia_smi:
            self._notes.append("No psutil/procfs/nvidia-smi detected; resource sampling disabled.")
            return

        if self._psutil is not None:
            try:
                self._psutil.cpu_percent(interval=None)
            except Exception:
                pass
        if self._has_proc:
            try:
                total, idle = self._read_proc_cpu_times()
                self._prev_cpu_total = total
                self._prev_cpu_idle = idle
            except Exception as exc:
                self._notes.append(f"Failed to init /proc CPU sampling: {exc}")
                self._has_proc = False

        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=max(5.0, self.interval_s * 2.0))

    def summary(self) -> Dict[str, Any]:
        cpu_vals = [s["cpu_percent"] for s in self._samples if isinstance(s.get("cpu_percent"), (int, float))]
        ram_vals = [s["ram_used_gb"] for s in self._samples if isinstance(s.get("ram_used_gb"), (int, float))]
        gpu_vals = [s["gpu_util_percent"] for s in self._samples if isinstance(s.get("gpu_util_percent"), (int, float))]
        gpu_mem_vals = [
            s["gpu_mem_used_mb"] for s in self._samples if isinstance(s.get("gpu_mem_used_mb"), (int, float))
        ]

        return {
            "sample_interval_sec": self.interval_s,
            "sample_count": len(self._samples),
            "cpu_percent_avg": _mean([float(v) for v in cpu_vals]),
            "cpu_percent_peak": float(max(cpu_vals)) if cpu_vals else None,
            "ram_used_gb_avg": _mean([float(v) for v in ram_vals]),
            "ram_used_gb_peak": float(max(ram_vals)) if ram_vals else None,
            "gpu_util_percent_avg": _mean([float(v) for v in gpu_vals]),
            "gpu_util_percent_peak": float(max(gpu_vals)) if gpu_vals else None,
            "gpu_mem_used_mb_avg": _mean([float(v) for v in gpu_mem_vals]),
            "gpu_mem_used_mb_peak": float(max(gpu_mem_vals)) if gpu_mem_vals else None,
            "notes": self._notes,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            sample: Dict[str, Any] = {"ts": time.time()}

            cpu_percent = self._sample_cpu_percent()
            if cpu_percent is not None:
                sample["cpu_percent"] = cpu_percent

            ram_used_gb = self._sample_ram_used_gb()
            if ram_used_gb is not None:
                sample["ram_used_gb"] = ram_used_gb

            gpu = self._sample_gpu()
            if gpu is not None:
                sample.update(gpu)

            self._samples.append(sample)
            self._stop.wait(self.interval_s)

    def _sample_cpu_percent(self) -> Optional[float]:
        if self._psutil is not None:
            try:
                return float(self._psutil.cpu_percent(interval=None))
            except Exception:
                return None
        if self._has_proc:
            try:
                total, idle = self._read_proc_cpu_times()
                if self._prev_cpu_total is None or self._prev_cpu_idle is None:
                    self._prev_cpu_total = total
                    self._prev_cpu_idle = idle
                    return None
                delta_total = total - self._prev_cpu_total
                delta_idle = idle - self._prev_cpu_idle
                self._prev_cpu_total = total
                self._prev_cpu_idle = idle
                if delta_total <= 0:
                    return None
                busy = max(0, delta_total - delta_idle)
                return float(busy / delta_total * 100.0)
            except Exception:
                return None
        return None

    def _sample_ram_used_gb(self) -> Optional[float]:
        if self._psutil is not None:
            try:
                used = float(self._psutil.virtual_memory().used)
                return used / (1024.0**3)
            except Exception:
                return None
        if self._has_proc:
            try:
                used_bytes = self._read_proc_mem_used_bytes()
                return float(used_bytes) / (1024.0**3)
            except Exception:
                return None
        return None

    def _sample_gpu(self) -> Optional[Dict[str, Any]]:
        if not self._has_nvidia_smi:
            return None
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
        except Exception as exc:
            self._notes.append(f"nvidia-smi failed: {exc}")
            self._has_nvidia_smi = False
            return None

        if not out:
            return None

        utils: List[float] = []
        mems: List[float] = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                utils.append(float(parts[0]))
                mems.append(float(parts[1]))
            except ValueError:
                continue

        if not utils and not mems:
            return None

        return {
            "gpu_util_percent": _mean(utils) if utils else None,
            "gpu_mem_used_mb": float(sum(mems)) if mems else None,
        }

    def _read_proc_cpu_times(self) -> Tuple[int, int]:
        text = Path("/proc/stat").read_text(encoding="utf-8")
        first = text.splitlines()[0].split()
        if not first or first[0] != "cpu":
            raise ValueError("unexpected /proc/stat format")
        nums = [int(v) for v in first[1:]]
        if len(nums) < 4:
            raise ValueError("unexpected /proc/stat cpu fields")
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = int(sum(nums))
        return total, idle

    def _read_proc_mem_used_bytes(self) -> int:
        mem_total_kb = None
        mem_available_kb = None
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_available_kb = int(line.split()[1])
            if mem_total_kb is not None and mem_available_kb is not None:
                break
        if mem_total_kb is None or mem_available_kb is None:
            raise ValueError("missing MemTotal/MemAvailable in /proc/meminfo")
        used_kb = mem_total_kb - mem_available_kb
        return int(used_kb * 1024)


@dataclass
class ExperimentConfig:
    name: str
    distributions: List[str]
    request_numbers: List[int]
    algorithms: List[str]
    seeds: List[int]
    generator_workers: int = 1
    max_workers: Optional[int] = None
    run_baseline: bool = True
    baseline_include_random: bool = False
    run_plots: bool = True
    run_metrics: bool = True
    cleanup_after_run: bool = False
    # Explicitly exclude specific (dist, request_number, algorithm, seed) tasks.
    # Useful for resuming partial runs without rerunning completed combinations.
    exclude_tasks: Optional[List[Tuple[str, int, str, int]]] = None
    # Robustness controls.
    resume_existing: bool = True
    skip_completed: bool = True
    notify_on_failure: bool = True
    notify_on_success: bool = False
    # Output root under codes/logs. If absolute path is provided, it will be used directly.
    log_subdir: str = ""
    # Algorithm version tag for extensible entries (e.g., PPO_NEW v1/v2/...).
    algo_version: str = "v1"
    # Optional PPO_NEW stacked window size; when set, pass to master via RL_PPO_NEW_WINDOW.
    ppo_new_window: Optional[int] = None
    # train_eval/train_only/eval_only
    stage_mode: str = "train_eval"
    # Optional checkpoint I/O passthrough to master.
    init_model_path: Optional[str] = None
    save_model_path: Optional[str] = None


@dataclass
class TaskPlan:
    dist_name: str
    request_number: int
    algorithm: str
    seed: int
    run_name: str
    run_dir: Path
    mode: str  # "new" | "resume"
    source_run: Optional[str] = None


class RunLease:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.lock_path = run_dir / LOCK_FILE
        self.heartbeat_path = run_dir / HEARTBEAT_FILE
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.token = (
            f"{self.hostname}:{self.pid}:"
            f"{threading.get_ident()}:{int(time.time() * 1000)}"
        )
        self.lease_s = max(30.0, _env_float("RUN_LOCK_LEASE_S", 900.0))
        self.heartbeat_s = max(3.0, _env_float("RUN_HEARTBEAT_S", 15.0))
        self.acquire_timeout_s = max(0.0, _env_float("RUN_LOCK_ACQUIRE_TIMEOUT_S", 120.0))
        self.steal_on_stale = _env_bool("RUN_LOCK_STEAL_STALE", True)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.acquired = False

    def acquire(self) -> bool:
        deadline = time.monotonic() + self.acquire_timeout_s
        while True:
            if self._try_create_lock():
                self.acquired = True
                self._write_heartbeat("running")
                self._thread = threading.Thread(target=self._heartbeat_loop, name="run-lease-heartbeat", daemon=True)
                self._thread.start()
                return True

            data = _load_json_file(self.lock_path) or {}
            if self.steal_on_stale and self._is_stale(data):
                self._break_stale_lock(data)
                continue

            if time.monotonic() >= deadline:
                return False
            time.sleep(min(5.0, self.heartbeat_s))

    def release(self, final_status: str) -> None:
        if not self.acquired:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(3.0, self.heartbeat_s * 2.0))
        self._refresh_lock(final_status)
        self._write_heartbeat(final_status)
        self._delete_lock_if_owned()
        self.acquired = False

    def _try_create_lock(self) -> bool:
        payload = {
            "status": "running",
            "token": self.token,
            "hostname": self.hostname,
            "pid": self.pid,
            "acquired_ts": time.time(),
            "last_heartbeat_ts": time.time(),
            "lease_s": self.lease_s,
        }
        try:
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            fd = os.open(str(self.lock_path), flags)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, indent=2))
            return True
        except FileExistsError:
            return False
        except Exception:
            return False

    def _is_stale(self, data: Dict[str, Any]) -> bool:
        now = time.time()
        last_ts = data.get("last_heartbeat_ts", data.get("acquired_ts", 0))
        try:
            last_ts_f = float(last_ts)
        except Exception:
            return True
        return (now - last_ts_f) > self.lease_s

    def _break_stale_lock(self, data: Dict[str, Any]) -> None:
        suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stale_path = self.run_dir / f"{LOCK_FILE}.stale.{suffix}"
        try:
            self.lock_path.replace(stale_path)
        except Exception:
            _safe_unlink(self.lock_path)
        _append_watchdog_event(
            self.run_dir,
            {
                "stage": "lock",
                "event": "stale_lock_takeover",
                "previous": data,
                "new_owner": {"hostname": self.hostname, "pid": self.pid, "token": self.token},
            },
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            self._refresh_lock("running")
            self._write_heartbeat("running")
            self._stop.wait(self.heartbeat_s)

    def _refresh_lock(self, status: str) -> None:
        payload = {
            "status": status,
            "token": self.token,
            "hostname": self.hostname,
            "pid": self.pid,
            "lease_s": self.lease_s,
            "last_heartbeat_ts": time.time(),
        }
        try:
            _atomic_write_json(self.lock_path, payload)
        except Exception:
            pass

    def _write_heartbeat(self, status: str) -> None:
        payload = {
            "status": status,
            "token": self.token,
            "hostname": self.hostname,
            "pid": self.pid,
            "ts": time.time(),
        }
        try:
            _atomic_write_json(self.heartbeat_path, payload)
        except Exception:
            pass

    def _delete_lock_if_owned(self) -> None:
        data = _load_json_file(self.lock_path) or {}
        if data.get("token") != self.token:
            return
        _safe_unlink(self.lock_path)


class NotificationManager:
    def __init__(self, run_root: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self.hostname = socket.gethostname()
        self.webhook_url = os.environ.get("EXP_NOTIFY_WEBHOOK_URL", "").strip()
        self.cooldown_s = max(0, _env_int("EXP_NOTIFY_COOLDOWN_S", 900))
        self.run_root = run_root or LOG_ROOT
        self.cooldown_file = self.run_root / "notify_cooldown.json"
        self.cooldown_on_send_fail = _env_bool("EXP_NOTIFY_COOLDOWN_ON_SEND_FAIL", True)

        self.smtp_host = os.environ.get("EXP_NOTIFY_SMTP_HOST", "").strip()
        self.smtp_port = _env_int("EXP_NOTIFY_SMTP_PORT", 587)
        self.smtp_user = os.environ.get("EXP_NOTIFY_SMTP_USER", "").strip()
        self.smtp_password = os.environ.get("EXP_NOTIFY_SMTP_PASSWORD", "").strip()
        self.smtp_from = os.environ.get("EXP_NOTIFY_SMTP_FROM", self.smtp_user or "").strip()
        self.smtp_to = os.environ.get("EXP_NOTIFY_SMTP_TO", "").strip()
        self.smtp_use_ssl = _env_bool("EXP_NOTIFY_SMTP_SSL", False)
        self.smtp_use_tls = _env_bool("EXP_NOTIFY_SMTP_TLS", not self.smtp_use_ssl)

        self.twilio_sid = os.environ.get("EXP_NOTIFY_TWILIO_ACCOUNT_SID", "").strip()
        self.twilio_token = os.environ.get("EXP_NOTIFY_TWILIO_AUTH_TOKEN", "").strip()
        self.twilio_from = os.environ.get("EXP_NOTIFY_TWILIO_FROM", "").strip()
        self.twilio_to = os.environ.get("EXP_NOTIFY_TWILIO_TO", "").strip()

    @property
    def channels(self) -> List[str]:
        result: List[str] = []
        if self.webhook_url:
            result.append("webhook")
        if self.smtp_host and self.smtp_to and self.smtp_from:
            result.append("email")
        if self.twilio_sid and self.twilio_token and self.twilio_from and self.twilio_to:
            result.append("twilio_sms")
        return result

    @property
    def enabled(self) -> bool:
        return bool(self.channels)

    def send(self, event: str, title: str, message: str, payload: Optional[Dict[str, Any]] = None) -> bool:
        if not self.enabled:
            return False
        payload_dict = dict(payload or {})
        key = self._make_cooldown_key(event, title, payload_dict)

        body_lines = [
            f"event={event}",
            f"host={self.hostname}",
            f"time={datetime.datetime.now().isoformat(timespec='seconds')}",
            "",
            message,
        ]
        body = "\n".join(body_lines).strip()
        ok = False
        with self._lock:
            if self._in_cooldown(key):
                print(f"{_c('[notify]', 'yellow', True)} cooldown skip: {key}")
                return False
            if self.webhook_url:
                ok = self._send_webhook(event, title, body, payload_dict) or ok
            if self.smtp_host and self.smtp_to and self.smtp_from:
                ok = self._send_email(title, body) or ok
            if self.twilio_sid and self.twilio_token and self.twilio_from and self.twilio_to:
                sms_body = f"{title}\n{message}"
                if len(sms_body) > 1400:
                    sms_body = sms_body[:1397] + "..."
                ok = self._send_twilio_sms(sms_body) or ok
            if ok or self.cooldown_on_send_fail:
                self._mark_cooldown(key)
        return ok

    def _make_cooldown_key(self, event: str, title: str, payload: Dict[str, Any]) -> str:
        run_name = str(payload.get("run_name", "")).strip()
        stage = str(payload.get("stage", "")).strip()
        status = str(payload.get("status", "")).strip()
        if run_name:
            return f"{event}:{run_name}:{stage}:{status}"
        return f"{event}:{title}"

    def _in_cooldown(self, key: str) -> bool:
        if self.cooldown_s <= 0:
            return False
        data = _load_json_file(self.cooldown_file) or {}
        last_ts = data.get(key)
        if last_ts is None:
            return False
        try:
            last_val = float(last_ts)
        except Exception:
            return False
        return (time.time() - last_val) < float(self.cooldown_s)

    def _mark_cooldown(self, key: str) -> None:
        if self.cooldown_s <= 0:
            return
        data = _load_json_file(self.cooldown_file) or {}
        data[key] = time.time()
        if len(data) > 2000:
            # Keep cooldown map bounded.
            keep = sorted(data.items(), key=lambda item: float(item[1]), reverse=True)[:1000]
            data = {k: v for k, v in keep}
        try:
            _atomic_write_json(self.cooldown_file, data)
        except Exception:
            pass

    def _send_webhook(self, event: str, title: str, body: str, payload: Optional[Dict[str, Any]]) -> bool:
        try:
            data = {
                "event": event,
                "title": title,
                "message": body,
                "hostname": self.hostname,
                "payload": payload or {},
            }
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(self.webhook_url, data=raw, method="POST")
            req.add_header("Content-Type", "application/json; charset=utf-8")
            with urllib.request.urlopen(req, timeout=20) as resp:
                status = getattr(resp, "status", 200)
            return int(status) < 400
        except Exception as exc:
            print(f"{_c('[notify]', 'red', True)} webhook failed: {exc}")
            return False

    def _send_email(self, subject: str, body: str) -> bool:
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = self.smtp_from
            msg["To"] = self.smtp_to
            msg.set_content(body)

            if self.smtp_use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=20)
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20)
            with server:
                if not self.smtp_use_ssl and self.smtp_use_tls:
                    server.starttls()
                if self.smtp_user:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            return True
        except Exception as exc:
            print(f"{_c('[notify]', 'red', True)} email failed: {exc}")
            return False

    def _send_twilio_sms(self, body: str) -> bool:
        try:
            endpoint = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_sid}/Messages.json"
            data = urllib.parse.urlencode(
                {"From": self.twilio_from, "To": self.twilio_to, "Body": body}
            ).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, method="POST")
            token = base64.b64encode(f"{self.twilio_sid}:{self.twilio_token}".encode("utf-8")).decode("ascii")
            req.add_header("Authorization", f"Basic {token}")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            with urllib.request.urlopen(req, timeout=20) as resp:
                status = getattr(resp, "status", 200)
            return int(status) < 400
        except Exception as exc:
            print(f"{_c('[notify]', 'red', True)} twilio failed: {exc}")
            return False


def detect_physical_cores() -> int:
    try:
        import psutil

        count = psutil.cpu_count(logical=False)
        if count:
            return count
    except Exception:
        pass
    return os.cpu_count() or 1


def resolve_max_workers(config: ExperimentConfig, override: Optional[int]) -> int:
    if override is not None and override > 0:
        return override
    if config.max_workers is not None and config.max_workers > 0:
        return config.max_workers
    reserved = max(1, _env_int("EXP_RESERVED_CORES", 2))
    return max(1, detect_physical_cores() - reserved)


def _normalize_algo_version_tag(version: Optional[str]) -> str:
    value = str(version or "").strip().lower()
    if not value:
        return "v1"
    # keep alnum and separators only, normalize for folder safety/readability
    cleaned = []
    for ch in value:
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in (".", "-", ":"):
            cleaned.append("_")
        elif ch == "_":
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    return out or "v1"


def _algorithm_run_tag(algorithm: str, algo_version: Optional[str]) -> str:
    algo = str(algorithm or "").strip().upper()
    if algo == "PPO_NEW":
        v = _normalize_algo_version_tag(algo_version).upper()
        return f"PPONEW{v}"
    return algo


def build_run_name(
    dist_name: str, request_number: int, algorithm: str, seed: Optional[int], algo_version: Optional[str] = None
) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    seed_tag = f"S{seed}" if seed is not None else "SNA"
    algo_tag = _algorithm_run_tag(algorithm, algo_version)
    return f"run_{timestamp}_R{request_number}_{dist_name}_{algo_tag}_{seed_tag}"


def resolve_run_root(config: ExperimentConfig) -> Path:
    candidate = str(getattr(config, "log_subdir", "") or "").strip()
    if not candidate:
        return LOG_ROOT
    path = Path(candidate)
    if path.is_absolute():
        return path
    return LOG_ROOT / path


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout_s: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return 124
    return proc.returncode


def _write_run_status(run_dir: Path, payload: Dict[str, Any]) -> None:
    try:
        payload = dict(payload)
        payload.setdefault("ts", time.time())
        (run_dir / "run_status.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _append_watchdog_event(run_dir: Path, payload: Dict[str, Any]) -> None:
    try:
        payload = dict(payload)
        payload.setdefault("ts", time.time())
        with (run_dir / "watchdog_events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _write_failed_marker(
    run_dir: Path, stage: str, code: Optional[int] = None, extra: Optional[Dict[str, Any]] = None
) -> None:
    try:
        payload: Dict[str, Any] = {"stage": stage, "status": "failed"}
        if code is not None:
            payload["exit_code"] = int(code)
        if extra:
            payload.update(extra)
        payload["ts"] = time.time()
        (run_dir / "FAILED.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _rotate_files(run_dir: Path, file_names: Iterable[str], attempt: int) -> None:
    for name in file_names:
        path = run_dir / name
        if not path.exists():
            continue
        try:
            path.unlink()
        except Exception:
            _safe_unlink(path)


def _rotate_master_files(run_dir: Path, attempt: int) -> None:
    _rotate_files(run_dir, MASTER_OUTPUT_FILES, attempt)
    _safe_unlink(run_dir / "34959.txt")


def _baseline_required_policies(include_random: bool) -> List[str]:
    policies = ["wait", "reroute"]
    if include_random:
        policies.append("random")
    return policies


def _baseline_policy_filename(policy: str) -> str:
    value = str(policy).strip().lower()
    if value == "wait":
        return "baseline_wait.csv"
    if value == "reroute":
        return "baseline_reroute.csv"
    if value == "random":
        return "baseline_random.csv"
    raise ValueError(f"unknown baseline policy: {policy}")


def _baseline_policy_path(run_dir: Path, policy: str) -> Path:
    return run_dir / _baseline_policy_filename(policy)


def _baseline_presence_flags(run_dir: Path) -> Dict[str, bool]:
    wait_path = _baseline_policy_path(run_dir, "wait")
    reroute_path = _baseline_policy_path(run_dir, "reroute")
    random_path = _baseline_policy_path(run_dir, "random")
    wait_file = _csv_has_data_row(wait_path, min_rows=1)
    reroute_file = _csv_has_data_row(reroute_path, min_rows=1)
    random_file = _csv_has_data_row(random_path, min_rows=1)
    return {
        "wait": wait_file,
        "reroute": reroute_file,
        "random": random_file,
        "wait_file": wait_file,
        "reroute_file": reroute_file,
        "random_file": random_file,
        "wait_impl": _csv_has_phase_row(wait_path, phase="implement", min_rows=1),
        "reroute_impl": _csv_has_phase_row(reroute_path, phase="implement", min_rows=1),
        "random_impl": _csv_has_phase_row(random_path, phase="implement", min_rows=1),
        "paper": _dir_has_content(run_dir / "paper_figures"),
    }


def _baseline_success_flags(run_dir: Path) -> Dict[str, bool]:
    flags = _baseline_presence_flags(run_dir)
    # Avoid baseline/plot cyclic dependency: baseline success should not require paper.
    wait_success = bool(flags["reroute"] or flags["wait_impl"] or (flags["paper"] and flags["wait"]))
    reroute_success = bool(
        flags["random"] or flags["reroute_impl"] or (flags["paper"] and flags["wait"] and flags["reroute"])
    )
    random_success = bool(flags["random_impl"] or (flags["paper"] and flags["wait"] and flags["reroute"] and flags["random"]))
    return {
        "wait": wait_success,
        "reroute": reroute_success,
        "random": random_success,
        "paper": bool(flags["paper"]),
        "wait_file": bool(flags["wait_file"]),
        "reroute_file": bool(flags["reroute_file"]),
        "random_file": bool(flags["random_file"]),
        "wait_impl": bool(flags["wait_impl"]),
        "reroute_impl": bool(flags["reroute_impl"]),
        "random_impl": bool(flags["random_impl"]),
    }


def _missing_baseline_policies(run_dir: Path, include_random: bool) -> List[str]:
    success = _baseline_success_flags(run_dir)
    missing: List[str] = []
    for policy in _baseline_required_policies(include_random):
        if not bool(success.get(policy, False)):
            missing.append(policy)
    return missing


def _build_baseline_cmd(run_dir: Path, policy: str) -> List[str]:
    return [
        sys.executable,
        str(CODES_DIR / "experiments" / "run_benchmark_replay.py"),
        "--run-dir",
        str(run_dir),
        "--policy",
        str(policy),
    ]


def _rotate_baseline_policy_file(run_dir: Path, policy: str, attempt: int) -> None:
    path = _baseline_policy_path(run_dir, policy)
    if not path.exists():
        return
    try:
        path.unlink()
    except Exception:
        _safe_unlink(path)


def _cleanup_attempt_files(run_dir: Path, file_names: Iterable[str]) -> None:
    for name in file_names:
        for path in run_dir.glob(name + ".attempt*"):
            _safe_unlink(path)


def _cleanup_master_attempts(run_dir: Path) -> None:
    _cleanup_attempt_files(run_dir, MASTER_OUTPUT_FILES)


def _resolve_master_watch_files(run_dir: Path) -> List[Path]:
    return [
        run_dir / "console_output.txt",
        run_dir / "rl_trace.csv",
        run_dir / "rl_training.csv",
    ]


def _resolve_baseline_watch_files(run_dir: Path, policy: Optional[str] = None) -> List[Path]:
    if policy:
        return [_baseline_policy_path(run_dir, policy)]
    return [_baseline_policy_path(run_dir, p) for p in ["wait", "reroute", "random"]]


def _cleanup_baseline_attempts(run_dir: Path) -> None:
    _cleanup_attempt_files(run_dir, BASELINE_OUTPUT_FILES)


def _run_with_watchdog(
    cmd: List[str],
    cwd: Optional[Path],
    *,
    run_dir: Path,
    stage: str,
    watch_files: List[Path],
    stall_s: float,
    startup_s: float,
    min_bytes: int,
    poll_s: float = 5.0,
    env: Optional[Dict[str, str]] = None,
) -> int:
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env)
    _write_run_status(run_dir, {"stage": stage, "status": "running", "pid": proc.pid})
    _append_watchdog_event(run_dir, {"stage": stage, "event": "start", "pid": proc.pid})

    last_sizes: Dict[Path, int] = {p: -1 for p in watch_files}
    last_growth_ts = time.monotonic()
    start_ts = last_growth_ts

    while True:
        code = proc.poll()
        if code is not None:
            _write_run_status(run_dir, {"stage": stage, "status": "finished", "exit_code": code})
            _append_watchdog_event(run_dir, {"stage": stage, "event": "exit", "code": code})
            return int(code)

        now = time.monotonic()
        max_size = 0
        for path in watch_files:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                size = 0
            max_size = max(max_size, size)
            prev = last_sizes.get(path, -1)
            if size > prev:
                last_sizes[path] = size
                last_growth_ts = now

        if now - start_ts >= startup_s and now - last_growth_ts >= stall_s:
            _append_watchdog_event(
                run_dir,
                {
                    "stage": stage,
                    "event": "stall_timeout",
                    "reason": "no_output" if max_size <= min_bytes else "no_growth",
                    "stall_s": stall_s,
                    "startup_s": startup_s,
                    "min_bytes": min_bytes,
                    "max_size": max_size,
                },
            )
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            _write_run_status(run_dir, {"stage": stage, "status": "killed", "reason": "stall_timeout"})
            return 124

        time.sleep(poll_s)


def _classify_exit_reason(code: int) -> str:
    if int(code) == 124:
        return "timeout"
    if int(code) in {130, 137, 143}:
        return "interrupted"
    return "nonzero_exit"


def _resolve_retry_budgets(stage: str, default_retries: int) -> Dict[str, int]:
    stage_key = stage.upper()
    timeout_retries = max(0, _env_int(f"{stage_key}_RETRIES_TIMEOUT", default_retries))
    nonzero_default = max(0, default_retries // 2)
    nonzero_retries = max(0, _env_int(f"{stage_key}_RETRIES_NONZERO", nonzero_default))
    interrupted_retries = max(0, _env_int(f"{stage_key}_RETRIES_INTERRUPTED", 0))
    return {
        "timeout": timeout_retries,
        "nonzero_exit": nonzero_retries,
        "interrupted": interrupted_retries,
    }


def _can_retry(reason: str, attempt: int, budgets: Dict[str, int]) -> bool:
    retries = int(budgets.get(reason, 0))
    # attempt starts at 1. retries means extra retries after first attempt.
    return attempt <= retries


def _retry_backoff_seconds(stage: str, attempt: int) -> float:
    base = max(0.0, _env_float("RETRY_BACKOFF_BASE_S", 5.0))
    cap = max(base, _env_float("RETRY_BACKOFF_MAX_S", 120.0))
    per_stage_mult = _env_float(f"{stage.upper()}_RETRY_BACKOFF_MULT", 1.0)
    delay = base * (2 ** max(0, attempt - 1)) * max(0.1, per_stage_mult)
    return min(cap, delay)


def _is_master_complete(run_dir: Path) -> bool:
    trace_ok = _csv_has_data_row(run_dir / "rl_trace.csv", min_rows=max(1, _env_int("MASTER_MIN_TRACE_ROWS", 5)))
    train_ok = _csv_has_data_row(
        run_dir / "rl_training.csv", min_rows=max(1, _env_int("MASTER_MIN_TRAIN_ROWS", 5))
    )
    summary_ok = _csv_has_data_row(
        run_dir / "rl_summary.csv", min_rows=max(1, _env_int("MASTER_MIN_SUMMARY_ROWS", 1))
    )
    baseline_any = any(
        _csv_has_data_row(run_dir / name, min_rows=1)
        for name in ("baseline_wait.csv", "baseline_reroute.csv", "baseline_random.csv")
    )
    return trace_ok and train_ok and (summary_ok or baseline_any)


def _is_baseline_complete(run_dir: Path, include_random: bool) -> bool:
    return len(_missing_baseline_policies(run_dir, include_random)) == 0


def _is_metrics_complete(run_dir: Path) -> bool:
    return _file_has_content(run_dir / "metrics.json", min_bytes=16)


def _is_plots_complete(run_dir: Path) -> bool:
    return _dir_has_content(run_dir / "paper_figures")


def _is_cleanup_complete(run_dir: Path) -> bool:
    return not (run_dir / "data").exists() and not (run_dir / "alns_outputs").exists()


def _collect_stage_state(run_dir: Path, config: ExperimentConfig) -> Dict[str, bool]:
    state: Dict[str, bool] = {
        "master": _is_master_complete(run_dir),
        "baseline": True,
        "metrics": True,
        "plot": True,
    }
    if config.run_baseline:
        state["baseline"] = _is_baseline_complete(run_dir, config.baseline_include_random)
    if config.run_metrics:
        state["metrics"] = _is_metrics_complete(run_dir)
    if config.run_plots:
        state["plot"] = _is_plots_complete(run_dir)
    return state


def _is_run_complete(run_dir: Path, config: ExperimentConfig) -> bool:
    if (run_dir / "FAILED.json").exists():
        return False
    done_path = _done_marker_path(run_dir)
    has_done = _file_has_content(done_path, min_bytes=2)
    state = _collect_stage_state(run_dir, config)
    all_complete = all(state.values())
    if has_done and all_complete:
        return True
    if has_done and not all_complete:
        # Marker is stale compared to actual files.
        _clear_done_marker(run_dir)
        return False
    if all_complete:
        # Backfill for old runs without marker.
        _write_done_marker(run_dir, {"source": "legacy_backfill", "state": state})
        return True
    return False


def _extract_run_key_from_meta(run_dir: Path) -> Optional[Tuple[str, int, str, int]]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    try:
        dist_name = str(data.get("distribution", "")).strip()
        request_number = int(data.get("request_number"))
        algorithm = str(data.get("algorithm", "")).strip()
        seed = int(data.get("seed"))
    except Exception:
        return None
    if not dist_name or not algorithm:
        return None
    return dist_name, request_number, algorithm, seed


def _parse_run_name_key(run_name: str, known_algorithms: Iterable[str]) -> Optional[Tuple[str, int, str, int]]:
    if not run_name.startswith("run_") or "_R" not in run_name or "_S" not in run_name:
        return None
    try:
        left, seed_raw = run_name.rsplit("_S", 1)
        seed = int(seed_raw)
        _, suffix = left.split("_R", 1)
        request_raw, rest = suffix.split("_", 1)
        request_number = int(request_raw)
    except Exception:
        return None

    algos = sorted({str(a).strip() for a in known_algorithms if str(a).strip()}, key=len, reverse=True)
    for algorithm in algos:
        marker = f"_{algorithm}"
        if marker in rest:
            idx = rest.rfind(marker)
            dist_name = rest[:idx]
            # supports suffixes like _PPO_NEW_V3_1, _PPONEWV3, etc.
            suffix = rest[idx + 1 :]
            if dist_name and str(suffix).startswith(algorithm):
                return dist_name, request_number, algorithm, seed
        if algorithm == "PPO_NEW":
            marker_new = "_PPONEW"
            if marker_new in rest:
                idx = rest.rfind(marker_new)
                dist_name = rest[:idx]
                if dist_name:
                    return dist_name, request_number, algorithm, seed
    return None


def _normalize_algo_version_for_match(version: Any) -> str:
    return _normalize_algo_version_tag(str(version or "").strip().lower())


def _run_matches_algo_version(run_dir: Path, algorithm: str, algo_version: str) -> bool:
    algo = str(algorithm or "").strip().upper()
    if algo != "PPO_NEW":
        return True
    target = _normalize_algo_version_for_match(algo_version)
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            run_ver = _normalize_algo_version_for_match(data.get("algo_version", "v1"))
            return run_ver == target
        except Exception:
            pass
    # Fallback by run name tag when meta is missing.
    name = run_dir.name.upper()
    needle = f"PPONEW{target.upper()}"
    return needle in name


def _extract_run_key(run_dir: Path, known_algorithms: Iterable[str]) -> Optional[Tuple[str, int, str, int]]:
    key = _extract_run_key_from_meta(run_dir)
    if key is not None:
        return key
    return _parse_run_name_key(run_dir.name, known_algorithms)


def _discover_existing_runs(
    run_root: Path,
    target_keys: Iterable[Tuple[str, int, str, int]],
    known_algorithms: Iterable[str],
) -> Dict[Tuple[str, int, str, int], List[Path]]:
    mapping: Dict[Tuple[str, int, str, int], List[Path]] = {}
    key_set = set(target_keys)
    if not run_root.exists():
        return mapping
    for run_dir in run_root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        key = _extract_run_key(run_dir, known_algorithms)
        if key is None or key not in key_set:
            continue
        mapping.setdefault(key, []).append(run_dir)

    for key, paths in mapping.items():
        paths.sort(key=_safe_mtime, reverse=True)
        mapping[key] = paths
    return mapping


def build_execution_plan(
    config: ExperimentConfig, tasks: List[Tuple[str, int, str, int]], run_root: Path
) -> Tuple[List[TaskPlan], int]:
    plans: List[TaskPlan] = []
    skipped_completed = 0
    existing = _discover_existing_runs(run_root, tasks, config.algorithms)

    for dist_name, request_number, algorithm, seed in tasks:
        key = (str(dist_name), int(request_number), str(algorithm), int(seed))
        candidates = existing.get(key, [])
        if str(algorithm).strip().upper() == "PPO_NEW":
            candidates = [
                p for p in candidates if _run_matches_algo_version(p, str(algorithm), getattr(config, "algo_version", "v1"))
            ]

        completed_dir = None
        for run_dir in candidates:
            if _is_run_complete(run_dir, config):
                completed_dir = run_dir
                break

        if completed_dir is not None and config.skip_completed:
            skipped_completed += 1
            print(
                f"{_c(f'[{config.name}]', 'cyan', True)} "
                f"{_c('SKIP', 'yellow', True)} completed {_format_task_key(key)} -> {completed_dir.name}"
            )
            continue

        if config.resume_existing and candidates:
            chosen = candidates[0]
            plans.append(
                TaskPlan(
                    dist_name=dist_name,
                    request_number=request_number,
                    algorithm=algorithm,
                    seed=seed,
                    run_name=chosen.name,
                    run_dir=chosen,
                    mode="resume",
                    source_run=chosen.name,
                )
            )
            continue

        run_name = build_run_name(
            dist_name,
            request_number,
            algorithm,
            seed,
            algo_version=getattr(config, "algo_version", "v1"),
        )
        run_dir = run_root / run_name
        plans.append(
            TaskPlan(
                dist_name=dist_name,
                request_number=request_number,
                algorithm=algorithm,
                seed=seed,
                run_name=run_name,
                run_dir=run_dir,
                mode="new",
                source_run=None,
            )
        )

    return plans, skipped_completed


def run_task(
    plan: TaskPlan,
    config: ExperimentConfig,
    dry_run: bool,
    notifier: Optional[NotificationManager] = None,
) -> Tuple[str, str]:
    dist_name = plan.dist_name
    request_number = int(plan.request_number)
    algorithm = plan.algorithm
    seed = int(plan.seed)
    run_name = plan.run_name
    run_dir = plan.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    master_cmd = [
        sys.executable,
        str(CODES_DIR / "Dynamic_master34959.py"),
        "--dist_name",
        dist_name,
        "--request_number",
        str(request_number),
        "--algorithm",
        algorithm,
        "--workers",
        str(config.generator_workers),
        "--seed",
        str(seed),
        "--run-name",
        run_name,
    ]
    if str(algorithm).strip().upper() == "PPO_NEW":
        master_cmd.extend(["--algo_version", str(getattr(config, "algo_version", "v1") or "v1")])
    stage_mode = str(getattr(config, "stage_mode", "train_eval") or "train_eval").strip()
    if stage_mode:
        master_cmd.extend(["--stage-mode", stage_mode])
    init_model_path = str(getattr(config, "init_model_path", "") or "").strip()
    if init_model_path:
        master_cmd.extend(["--init-model-path", init_model_path])
    save_model_path = str(getattr(config, "save_model_path", "") or "").strip()
    if save_model_path:
        master_cmd.extend(["--save-model-path", save_model_path])
    master_env = dict(os.environ)
    master_env["RL_LOG_ROOT"] = str(run_dir.parent)
    if str(algorithm).strip().upper() == "PPO_NEW":
        window_k = getattr(config, "ppo_new_window", None)
        if window_k is not None:
            try:
                master_env["RL_PPO_NEW_WINDOW"] = str(max(1, int(window_k)))
            except Exception:
                pass

    plot_cmd = [
        sys.executable,
        str(CODES_DIR / "plotting" / "plot_paper_figure.py"),
        "--run-dir",
        str(run_dir),
    ]

    metrics_cmd = [
        sys.executable,
        str(CODES_DIR / "analysis" / "compute_metrics.py"),
        "--run-dir",
        str(run_dir),
    ]

    cleanup_cmd = [
        sys.executable,
        str(CODES_DIR / "tools" / "cleanup_run.py"),
        "--run-dir",
        str(run_dir),
    ]

    print(
        f"{_c(f'[{config.name}]', 'cyan', True)} "
        f"{_c('START', 'green', True)} {run_name} mode={plan.mode}"
    )
    state = _collect_stage_state(run_dir, config)
    baseline_detail = _baseline_success_flags(run_dir) if config.run_baseline else {}
    if plan.mode == "resume":
        print(
            f"{_c(f'[{config.name}]', 'cyan', True)} {_c('RESUME', 'blue', True)} stage_state {run_name}: "
            f"master={int(state['master'])} baseline={int(state['baseline'])} "
            f"metrics={int(state['metrics'])} plot={int(state['plot'])}"
        )
        if config.run_baseline:
            print(
                f"{_c(f'[{config.name}]', 'cyan', True)} {_c('RESUME', 'blue', True)} baseline_flags {run_name}: "
                f"W={int(baseline_detail.get('wait_file', False))} "
                f"R={int(baseline_detail.get('reroute_file', False))} "
                f"N={int(baseline_detail.get('random_file', False))} "
                f"WI={int(baseline_detail.get('wait_impl', False))} "
                f"RI={int(baseline_detail.get('reroute_impl', False))} "
                f"NI={int(baseline_detail.get('random_impl', False))} "
                f"P={int(baseline_detail.get('paper', False))} | "
                f"wait_ok={int(baseline_detail.get('wait', False))} "
                f"reroute_ok={int(baseline_detail.get('reroute', False))} "
                f"random_ok={int(baseline_detail.get('random', False))}"
            )
        _append_watchdog_event(
            run_dir,
            {"stage": "resume", "event": "stage_state", "state": state, "baseline_detail": baseline_detail},
        )

    if dry_run:
        print(f"  {_c('MASTER', 'green', True)}:", " ".join(master_cmd))
        if config.run_baseline:
            missing_policies = _missing_baseline_policies(run_dir, config.baseline_include_random)
            if not missing_policies:
                print(f"  {_c('BASELINE', 'yellow', True)}: <skip, already complete>")
            else:
                for policy in missing_policies:
                    print(f"  {_c(f'BASELINE[{policy}]', 'yellow', True)}:", " ".join(_build_baseline_cmd(run_dir, policy)))
        if config.run_metrics:
            print(f"  {_c('METRICS', 'magenta', True)}:", " ".join(metrics_cmd))
        if config.run_plots:
            print(f"  {_c('PLOT', 'blue', True)}:", " ".join(plot_cmd))
        if config.cleanup_after_run:
            print(f"  {_c('CLEANUP', 'cyan', True)}:", " ".join(cleanup_cmd))
        return run_name, "dry_run"

    lease = RunLease(run_dir)
    if not lease.acquire():
        _append_watchdog_event(
            run_dir,
            {
                "stage": "lock",
                "event": "lock_busy_skip",
                "owner": _load_json_file(run_dir / LOCK_FILE) or {},
            },
        )
        print(f"{_c(f'[{config.name}]', 'cyan', True)} {_c('LOCK', 'yellow', True)} busy -> skip {run_name}")
        return run_name, "skipped_locked"

    state = _collect_stage_state(run_dir, config)
    if not all(state.values()):
        _clear_done_marker(run_dir)

    monitor = ResourceMonitor(interval_s=1.0)
    monitor.start()
    stage_times: Dict[str, float] = {}
    started_at = time.monotonic()
    status = "ok"
    failure_detail: Dict[str, Any] = {}

    master_stall_s = _env_float("MASTER_STALL_S", 1800.0)
    master_startup_s = _env_float("MASTER_STARTUP_S", 600.0)
    master_min_bytes = _env_int("MASTER_MIN_BYTES", 1024)
    master_retries = max(0, _env_int("MASTER_RETRIES", 2))
    master_poll_s = _env_float("MASTER_POLL_S", 5.0)
    master_retry_budgets = _resolve_retry_budgets("master", master_retries)

    baseline_stall_s = _env_float("BASELINE_STALL_S", 600.0)
    baseline_startup_s = _env_float("BASELINE_STARTUP_S", 300.0)
    baseline_min_bytes = _env_int("BASELINE_MIN_BYTES", 4096)
    baseline_retries = max(0, _env_int("BASELINE_RETRIES", 3))
    baseline_poll_s = _env_float("BASELINE_POLL_S", 5.0)
    baseline_retry_budgets = _resolve_retry_budgets("baseline", baseline_retries)

    metrics_timeout_s = _env_float("METRICS_TIMEOUT_S", 3600.0)
    metrics_retries = max(0, _env_int("METRICS_RETRIES", 1))
    metrics_retry_budgets = _resolve_retry_budgets("metrics", metrics_retries)
    plots_timeout_s = _env_float("PLOTS_TIMEOUT_S", 3600.0)
    plots_retries = max(0, _env_int("PLOTS_RETRIES", 1))
    plots_retry_budgets = _resolve_retry_budgets("plots", plots_retries)
    cleanup_timeout_s = _env_float("CLEANUP_TIMEOUT_S", 900.0)
    cleanup_retries = max(0, _env_int("CLEANUP_RETRIES", 0))
    cleanup_retry_budgets = _resolve_retry_budgets("cleanup", cleanup_retries)

    master_complete = state["master"]
    baseline_complete = state["baseline"]
    metrics_complete = state["metrics"]
    plot_complete = state["plot"]
    master_reran = False

    try:
        if config.skip_completed and _is_run_complete(run_dir, config):
            if config.cleanup_after_run and not _is_cleanup_complete(run_dir):
                attempt = 0
                code = 1
                last_reason = "nonzero_exit"
                while True:
                    attempt += 1
                    t0 = time.monotonic()
                    code = run_command(cleanup_cmd, cwd=ROOT_DIR, timeout_s=cleanup_timeout_s)
                    stage_times[f"cleanup_sec_attempt_{attempt}"] = time.monotonic() - t0
                    if code == 0:
                        break
                    last_reason = _classify_exit_reason(code)
                    _append_watchdog_event(
                        run_dir,
                        {
                            "stage": "cleanup",
                            "event": "attempt_failed",
                            "attempt": attempt,
                            "exit_code": code,
                            "reason": last_reason,
                            "source": "skip_completed",
                        },
                    )
                    if not _can_retry(last_reason, attempt, cleanup_retry_budgets):
                        break
                    backoff = _retry_backoff_seconds("cleanup", attempt)
                    _append_watchdog_event(
                        run_dir,
                        {
                            "stage": "cleanup",
                            "event": "retry_scheduled",
                            "attempt": attempt + 1,
                            "after_s": backoff,
                            "reason": last_reason,
                            "source": "skip_completed",
                        },
                    )
                    time.sleep(backoff)
                if code != 0:
                    status = "failed_cleanup"
                    failure_detail = {
                        "stage": "cleanup",
                        "attempts": attempt,
                        "retry_budgets": cleanup_retry_budgets,
                        "exit_code": code,
                        "reason": last_reason,
                        "source": "skip_completed",
                    }
                    _write_failed_marker(run_dir, "cleanup", code, failure_detail)
                    return run_name, status
                stage_times["cleanup_sec"] = sum(
                    value for key, value in stage_times.items() if key.startswith("cleanup_sec_attempt_")
                )
            status = "skipped_completed"
            return run_name, status

        if not master_complete:
            _clear_done_marker(run_dir)
            attempt = 0
            code = 1
            last_reason = "nonzero_exit"
            watch_files = _resolve_master_watch_files(run_dir)
            while True:
                attempt += 1
                _rotate_master_files(run_dir, attempt)
                _append_watchdog_event(
                    run_dir,
                    {
                        "stage": "master",
                        "event": "attempt_start",
                        "attempt": attempt,
                        "max_attempts": master_retries + 1,
                    },
                )
                t0 = time.monotonic()
                code = _run_with_watchdog(
                    master_cmd,
                    ROOT_DIR,
                    run_dir=run_dir,
                    stage="master",
                    watch_files=watch_files,
                    stall_s=master_stall_s,
                    startup_s=master_startup_s,
                    min_bytes=master_min_bytes,
                    poll_s=master_poll_s,
                    env=master_env,
                )
                stage_times[f"master_sec_attempt_{attempt}"] = time.monotonic() - t0
                if code == 0:
                    break
                last_reason = _classify_exit_reason(code)
                _append_watchdog_event(
                    run_dir,
                    {
                        "stage": "master",
                        "event": "attempt_failed",
                        "attempt": attempt,
                        "exit_code": code,
                        "reason": last_reason,
                    },
                )
                if not _can_retry(last_reason, attempt, master_retry_budgets):
                    break
                backoff = _retry_backoff_seconds("master", attempt)
                _append_watchdog_event(
                    run_dir,
                    {
                        "stage": "master",
                        "event": "retry_scheduled",
                        "attempt": attempt + 1,
                        "after_s": backoff,
                        "reason": last_reason,
                    },
                )
                time.sleep(backoff)
            if code != 0:
                status = "failed_master"
                failure_detail = {
                    "stage": "master",
                    "attempts": attempt,
                    "retry_budgets": master_retry_budgets,
                    "exit_code": code,
                    "reason": last_reason,
                }
                _write_failed_marker(run_dir, "master", code, failure_detail)
                return run_name, status
            master_complete = True
            master_reran = True
            baseline_complete = False
            metrics_complete = False
            plot_complete = False
            _safe_unlink(run_dir / "metrics.json")
            _safe_rmtree(run_dir / "paper_figures")

        if config.run_baseline and not baseline_complete:
            _clear_done_marker(run_dir)
            if master_reran:
                missing_policies = _baseline_required_policies(config.baseline_include_random)
            else:
                missing_policies = _missing_baseline_policies(run_dir, config.baseline_include_random)
            baseline_ran_any = False
            for policy in missing_policies:
                attempt = 0
                code = 1
                last_reason = "nonzero_exit"
                baseline_cmd = _build_baseline_cmd(run_dir, policy)
                watch_files = _resolve_baseline_watch_files(run_dir, policy=policy)
                while True:
                    attempt += 1
                    _rotate_baseline_policy_file(run_dir, policy, attempt)
                    _append_watchdog_event(
                        run_dir,
                        {
                            "stage": "baseline",
                            "event": "attempt_start",
                            "policy": policy,
                            "attempt": attempt,
                            "max_attempts": baseline_retries + 1,
                        },
                    )
                    t0 = time.monotonic()
                    code = _run_with_watchdog(
                        baseline_cmd,
                        ROOT_DIR,
                        run_dir=run_dir,
                        stage=f"baseline_{policy}",
                        watch_files=watch_files,
                        stall_s=baseline_stall_s,
                        startup_s=baseline_startup_s,
                        min_bytes=baseline_min_bytes,
                        poll_s=baseline_poll_s,
                    )
                    stage_times[f"baseline_{policy}_sec_attempt_{attempt}"] = time.monotonic() - t0
                    if code == 0:
                        break
                    last_reason = _classify_exit_reason(code)
                    _append_watchdog_event(
                        run_dir,
                        {
                            "stage": "baseline",
                            "event": "attempt_failed",
                            "policy": policy,
                            "attempt": attempt,
                            "exit_code": code,
                            "reason": last_reason,
                        },
                    )
                    if not _can_retry(last_reason, attempt, baseline_retry_budgets):
                        break
                    backoff = _retry_backoff_seconds("baseline", attempt)
                    _append_watchdog_event(
                        run_dir,
                        {
                            "stage": "baseline",
                            "event": "retry_scheduled",
                            "policy": policy,
                            "attempt": attempt + 1,
                            "after_s": backoff,
                            "reason": last_reason,
                        },
                    )
                    time.sleep(backoff)
                if code != 0:
                    status = "failed_baseline"
                    failure_detail = {
                        "stage": "baseline",
                        "policy": policy,
                        "attempts": attempt,
                        "retry_budgets": baseline_retry_budgets,
                        "exit_code": code,
                        "reason": last_reason,
                        "missing_policies": missing_policies,
                    }
                    _write_failed_marker(run_dir, "baseline", code, failure_detail)
                    return run_name, status
                baseline_ran_any = True
                stage_times[f"baseline_{policy}_sec"] = sum(
                    value for key, value in stage_times.items() if key.startswith(f"baseline_{policy}_sec_attempt_")
                )

            baseline_complete = len(_missing_baseline_policies(run_dir, config.baseline_include_random)) == 0
            if not baseline_complete:
                status = "failed_baseline"
                failure_detail = {
                    "stage": "baseline",
                    "reason": "incomplete_after_policy_runs",
                    "missing_policies": _missing_baseline_policies(run_dir, config.baseline_include_random),
                }
                _write_failed_marker(run_dir, "baseline", 1, failure_detail)
                return run_name, status

            if baseline_ran_any:
                metrics_complete = False
                plot_complete = False
                _safe_unlink(run_dir / "metrics.json")
                _safe_rmtree(run_dir / "paper_figures")

        if config.run_metrics and not metrics_complete:
            _clear_done_marker(run_dir)
            attempt = 0
            code = 1
            last_reason = "nonzero_exit"
            while True:
                attempt += 1
                _safe_unlink(run_dir / "metrics.json")
                t0 = time.monotonic()
                code = run_command(metrics_cmd, cwd=ROOT_DIR, timeout_s=metrics_timeout_s)
                stage_times[f"metrics_sec_attempt_{attempt}"] = time.monotonic() - t0
                if code == 0:
                    break
                last_reason = _classify_exit_reason(code)
                _append_watchdog_event(
                    run_dir,
                    {
                        "stage": "metrics",
                        "event": "attempt_failed",
                        "attempt": attempt,
                        "exit_code": code,
                        "reason": last_reason,
                    },
                )
                if not _can_retry(last_reason, attempt, metrics_retry_budgets):
                    break
                backoff = _retry_backoff_seconds("metrics", attempt)
                _append_watchdog_event(
                    run_dir,
                    {
                        "stage": "metrics",
                        "event": "retry_scheduled",
                        "attempt": attempt + 1,
                        "after_s": backoff,
                        "reason": last_reason,
                    },
                )
                time.sleep(backoff)
            if code != 0:
                status = "failed_metrics"
                failure_detail = {
                    "stage": "metrics",
                    "attempts": attempt,
                    "retry_budgets": metrics_retry_budgets,
                    "exit_code": code,
                    "reason": last_reason,
                }
                _write_failed_marker(run_dir, "metrics", code, failure_detail)
                return run_name, status
            stage_times["metrics_sec"] = sum(
                value for key, value in stage_times.items() if key.startswith("metrics_sec_attempt_")
            )
            metrics_complete = True

        if config.run_plots and not plot_complete:
            _clear_done_marker(run_dir)
            attempt = 0
            code = 1
            last_reason = "nonzero_exit"
            while True:
                attempt += 1
                _safe_rmtree(run_dir / "paper_figures")
                t0 = time.monotonic()
                code = run_command(plot_cmd, cwd=ROOT_DIR, timeout_s=plots_timeout_s)
                stage_times[f"plot_sec_attempt_{attempt}"] = time.monotonic() - t0
                if code == 0:
                    break
                last_reason = _classify_exit_reason(code)
                _append_watchdog_event(
                    run_dir,
                    {
                        "stage": "plot",
                        "event": "attempt_failed",
                        "attempt": attempt,
                        "exit_code": code,
                        "reason": last_reason,
                    },
                )
                if not _can_retry(last_reason, attempt, plots_retry_budgets):
                    break
                backoff = _retry_backoff_seconds("plots", attempt)
                _append_watchdog_event(
                    run_dir,
                    {
                        "stage": "plot",
                        "event": "retry_scheduled",
                        "attempt": attempt + 1,
                        "after_s": backoff,
                        "reason": last_reason,
                    },
                )
                time.sleep(backoff)
            if code != 0:
                status = "failed_plot"
                failure_detail = {
                    "stage": "plot",
                    "attempts": attempt,
                    "retry_budgets": plots_retry_budgets,
                    "exit_code": code,
                    "reason": last_reason,
                }
                _write_failed_marker(run_dir, "plot", code, failure_detail)
                return run_name, status
            stage_times["plot_sec"] = sum(
                value for key, value in stage_times.items() if key.startswith("plot_sec_attempt_")
            )
            plot_complete = True

        if config.cleanup_after_run:
            attempt = 0
            code = 1
            last_reason = "nonzero_exit"
            while True:
                attempt += 1
                t0 = time.monotonic()
                code = run_command(cleanup_cmd, cwd=ROOT_DIR, timeout_s=cleanup_timeout_s)
                stage_times[f"cleanup_sec_attempt_{attempt}"] = time.monotonic() - t0
                if code == 0:
                    break
                last_reason = _classify_exit_reason(code)
                _append_watchdog_event(
                    run_dir,
                    {
                        "stage": "cleanup",
                        "event": "attempt_failed",
                        "attempt": attempt,
                        "exit_code": code,
                        "reason": last_reason,
                    },
                )
                if not _can_retry(last_reason, attempt, cleanup_retry_budgets):
                    break
                backoff = _retry_backoff_seconds("cleanup", attempt)
                _append_watchdog_event(
                    run_dir,
                    {
                        "stage": "cleanup",
                        "event": "retry_scheduled",
                        "attempt": attempt + 1,
                        "after_s": backoff,
                        "reason": last_reason,
                    },
                )
                time.sleep(backoff)
            if code != 0:
                status = "failed_cleanup"
                failure_detail = {
                    "stage": "cleanup",
                    "attempts": attempt,
                    "retry_budgets": cleanup_retry_budgets,
                    "exit_code": code,
                    "reason": last_reason,
                }
                _write_failed_marker(run_dir, "cleanup", code, failure_detail)
                return run_name, status
            stage_times["cleanup_sec"] = sum(
                value for key, value in stage_times.items() if key.startswith("cleanup_sec_attempt_")
            )

        final_state = _collect_stage_state(run_dir, config)
        if not all(final_state.values()):
            status = "failed_postcheck"
            failure_detail = {
                "stage": "postcheck",
                "reason": "incomplete_outputs_after_success",
                "state": final_state,
            }
            _write_failed_marker(run_dir, "postcheck", 1, failure_detail)
            return run_name, status

        _write_done_marker(
            run_dir,
            {
                "run_name": run_name,
                "mode": plan.mode,
                "state": final_state,
                "stage_wall_time_sec": stage_times,
            },
        )

        return run_name, status
    finally:
        try:
            _cleanup_master_attempts(run_dir)
        except Exception:
            pass
        try:
            _cleanup_baseline_attempts(run_dir)
        except Exception:
            pass
        if status in {"ok", "dry_run", "skipped_completed"}:
            _safe_unlink(run_dir / "FAILED.json")
        if status.startswith("failed_"):
            _clear_done_marker(run_dir)
        try:
            monitor.stop()
            usage: Dict[str, Any] = monitor.summary()
            usage["status"] = status
            usage["mode"] = plan.mode
            usage["stage_wall_time_sec"] = stage_times
            usage["wall_time_sec"] = time.monotonic() - started_at
            if failure_detail:
                usage["failure"] = failure_detail
            (run_dir / "resource_usage.json").write_text(
                json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        try:
            lease.release(status)
        except Exception:
            pass

        if status.startswith("failed_"):
            print(
                f"{_c(f'[{config.name}]', 'cyan', True)} "
                f"{_c('RESULT', 'red' if status.startswith('failed_') else 'green', True)} "
                f"{run_name} -> {status} | detail={failure_detail}"
            )
            if notifier is not None and config.notify_on_failure:
                msg = "\n".join(
                    [
                        f"run_name={run_name}",
                        f"run_dir={run_dir}",
                        f"mode={plan.mode}",
                        f"status={status}",
                        f"detail={json.dumps(failure_detail, ensure_ascii=False)}",
                    ]
                )
                notifier.send(
                    event="task_failed",
                    title=f"[{config.name}] task failed: {run_name}",
                    message=msg,
                    payload={"run_name": run_name, "status": status, "failure_detail": failure_detail},
                )


def build_tasks(config: ExperimentConfig) -> List[Tuple[str, int, str, int]]:
    tasks: List[Tuple[str, int, str, int]] = []
    excluded = set()
    if config.exclude_tasks:
        for dist_name, request_number, algorithm, seed in config.exclude_tasks:
            excluded.add((str(dist_name), int(request_number), str(algorithm), int(seed)))
    for dist_name in config.distributions:
        for request_number in config.request_numbers:
            for algorithm in config.algorithms:
                for seed in config.seeds:
                    key = (str(dist_name), int(request_number), str(algorithm), int(seed))
                    if key in excluded:
                        continue
                    tasks.append((dist_name, request_number, algorithm, seed))
    return tasks


def run_experiments(config: ExperimentConfig, max_workers: Optional[int], dry_run: bool) -> int:
    tasks = build_tasks(config)
    if not tasks:
        print(_c("No tasks to run.", "yellow", True))
        return 1

    run_root = resolve_run_root(config)
    run_root.mkdir(parents=True, exist_ok=True)

    plans, skipped_completed = build_execution_plan(config, tasks, run_root)
    if not plans:
        print(f"{_c(f'[{config.name}]', 'cyan', True)} {_c('DONE', 'green', True)} all tasks already completed. skipped={skipped_completed}")
        notifier = NotificationManager(run_root=run_root)
        if notifier.enabled and config.notify_on_success:
            notifier.send(
                event="all_completed",
                title=f"[{config.name}] all tasks already completed",
                message=f"total={len(tasks)} skipped_completed={skipped_completed}",
                payload={"total": len(tasks), "skipped_completed": skipped_completed},
            )
        return 0

    worker_count = min(resolve_max_workers(config, max_workers), len(plans))
    notifier = NotificationManager(run_root=run_root)
    print(
        f"{_c(f'[{config.name}]', 'cyan', True)} {_c('QUEUE', 'blue', True)} "
        f"total={len(tasks)} queued={len(plans)} skipped_completed={skipped_completed} "
        f"workers={worker_count} run_root={run_root}"
    )
    if notifier.enabled:
        print(f"{_c(f'[{config.name}]', 'cyan', True)} {_c('NOTIFY', 'green', True)} enabled: {', '.join(notifier.channels)}")
    else:
        print(f"{_c(f'[{config.name}]', 'cyan', True)} {_c('NOTIFY', 'yellow', True)} disabled (set webhook/SMTP/Twilio env to enable)")

    failed = 0
    status_counter: Dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_task, plan, config, dry_run, notifier) for plan in plans]
        for future in concurrent.futures.as_completed(futures):
            try:
                run_name, status = future.result()
            except Exception:
                failed += 1
                status = "failed_internal_exception"
                status_counter[status] = status_counter.get(status, 0) + 1
                err = traceback.format_exc()
                print(f"{_c(f'[{config.name}]', 'cyan', True)} {_c('EXCEPTION', 'red', True)}\n{err}")
                if notifier.enabled and config.notify_on_failure:
                    notifier.send(
                        event="task_exception",
                        title=f"[{config.name}] internal exception",
                        message=err[-3000:],
                        payload={"status": status},
                    )
                continue

            status_counter[status] = status_counter.get(status, 0) + 1
            print(
                f"{_c(f'[{config.name}]', 'cyan', True)} "
                f"{_c('RESULT', 'red' if status.startswith('failed_') else 'green', True)} "
                f"{run_name} -> {status}"
            )
            if status not in {"ok", "dry_run", "skipped_completed", "skipped_locked"}:
                failed += 1

    summary_payload = {
        "total_tasks": len(tasks),
        "queued_tasks": len(plans),
        "skipped_completed": skipped_completed,
        "failed": failed,
        "status_counter": status_counter,
    }
    print(
        f"{_c(f'[{config.name}]', 'cyan', True)} {_c('SUMMARY', 'magenta', True)} "
        f"{json.dumps(summary_payload, ensure_ascii=False)}"
    )

    should_notify_summary = notifier.enabled and (
        (failed > 0 and config.notify_on_failure) or (failed == 0 and config.notify_on_success)
    )
    if should_notify_summary:
        notifier.send(
            event="run_summary",
            title=f"[{config.name}] summary failed={failed}",
            message=json.dumps(summary_payload, ensure_ascii=False, indent=2),
            payload=summary_payload,
        )

    return failed
