"""Supervise one llama-server process."""

from __future__ import annotations

import atexit
import http.client
import json
import os
import re
import secrets
import subprocess
import threading
import time
from pathlib import Path

import psutil

from .config import Config
from .discovery import IS_WINDOWS, ModelEntry, scrubbed_env, server_help, server_version
from .errors import BackendError
from .offline import ensure_port_free, free_loopback_port

# Known failure signatures in llama-server output -> (code, hint)
_DIAGNOSES = [
    (re.compile(r"error: invalid argument|unknown argument|error while handling argument", re.I), "backend_bad_argument",
     "This llama-server build does not accept one of the flags. Update llama.cpp (scripts/windows/fetch_runtime.ps1) "
     "or remove the flag from extra_args in runtime.toml."),
    (re.compile(r"unknown model architecture|unsupported model", re.I), "model_unsupported",
     "This llama.cpp build is too old for the model's architecture. Update llama.cpp or pick a different model."),
    (re.compile(r"failed to allocate|out of memory|std::bad_alloc|cannot allocate|mmap failed", re.I), "backend_out_of_memory",
     "Not enough memory. Pick a smaller quantization (Q4_K_M), lower ctx_size, or close other applications."),
    (re.compile(r"couldn't bind|address already in use|bind failed|only one usage of each socket", re.I), "backend_port_in_use",
     "The internal backend port is taken. Leave backend_port = 0 in runtime.toml to pick a free port automatically."),
    (re.compile(r"error loading model|failed to load model|invalid magic|tensor .* data is not within|corrupted", re.I), "model_load_failed",
     "The model file could not be loaded. It may be corrupt or incomplete - re-download it and compare the file size "
     "with the download page."),
    (re.compile(r"is not recognized|No such file|cannot execute|not a valid Win32", re.I), "backend_not_runnable",
     "llama-server could not be executed. Make sure you downloaded the Windows x64 CPU build and that all its DLLs "
     "are in the same folder."),
]

# Flags we would like to pass, and the token that must appear in --help for us to pass it.
_OPTIONAL_FLAGS = {"--offline": "--offline", "--no-webui": "--no-webui"}


_ERR_LINE = re.compile(r"(^|\s)E\s|error|failed|exception|terminate", re.I)


def diagnose(log_lines: list[str]) -> tuple[str, str]:
    text = "\n".join(log_lines)
    for rx, code, hint in _DIAGNOSES:
        if rx.search(text):
            return code, hint
    return "backend_exited", "llama-server stopped unexpectedly. Read the log lines above; the full log path is shown below."


# NB: PR_SET_PDEATHSIG is deliberately NOT used on Linux: it fires when the *thread* that spawned the child exits,
# which kills llama-server if the runtime is started from a worker thread (e.g. inside a desktop app).
_LIVE: "set[BackendProcess]" = set()


@atexit.register
def _stop_all_on_exit() -> None:
    for b in list(_LIVE):
        b.stop(timeout=5)


def _windows_kill_on_close_job(proc: subprocess.Popen):  # pragma: no cover - Windows only
    """Put the child in a Job Object so it dies if the launcher is killed."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class IOC(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class EXT(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IOC), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    info = EXT()
    info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # 9 = ExtendedLimitInformation
    k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle)))
    return job


class BackendProcess:
    def __init__(self, cfg: Config, model: ModelEntry, server_cmd: list[str], log_dir: Path):
        self.cfg = cfg
        self.model = model
        self.server_cmd = server_cmd
        self.log_dir = log_dir
        self.port = cfg.backend_port
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self._log_fh = None
        self._job = None
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self.args: list[str] = []
        self.version = "unknown"
        self.is_stub = False
        self.stopping = False
        self.api_key: str | None = None      # per-launch secret; only the gateway knows it

    # ---------------------------------------------------------- command --
    def build_args(self, help_text: str) -> list[str]:
        cfg = self.cfg
        threads = cfg.threads or psutil.cpu_count(logical=False) or os.cpu_count() or 4
        args = [
            "-m", str(self.model.path),
            "--alias", self.model.id,
            "--host", "127.0.0.1",                 # backend is never exposed; only the gateway is
            "--port", str(self.port),
            "-c", str(cfg.ctx_size),
            "-t", str(threads),
            "-ngl", str(cfg.n_gpu_layers),
        ]
        if not cfg.mmap:
            args.append("--no-mmap")
        if cfg.mlock:
            args.append("--mlock")
        for flag, token in _OPTIONAL_FLAGS.items():
            if flag == "--offline" and not cfg.offline:
                continue
            if token in help_text:
                args.append(flag)
        args += list(cfg.extra_args)
        return args

    @property
    def command(self) -> list[str]:
        return self.server_cmd + self.args

    def command_line(self) -> str:
        return subprocess.list2cmdline(self.command) if IS_WINDOWS else " ".join(_q(a) for a in self.command)

    # ------------------------------------------------------------ start --
    def start(self) -> None:
        if self.port:
            ensure_port_free("127.0.0.1", self.port, "backend_port", "backend_port in runtime.toml")
        else:
            self.port = free_loopback_port()
        help_text = server_help(self.server_cmd)
        self.version = server_version(self.server_cmd)
        self.is_stub = "docqa-stub" in self.version or "docqa stub" in help_text
        self.args = self.build_args(help_text)
        # Lock the backend: llama-server enables CORS for all origins by default, so without a key any web page
        # open in the user's browser could call the internal port directly and bypass the gateway's checks.
        # The key goes through the environment (not the command line) so it never appears in logs or run records.
        self.api_key = secrets.token_urlsafe(24) if "--api-key" in help_text else None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"llama-server-{time.strftime('%Y%m%d-%H%M%S')}-{self.model.id}.log"
        self._log_fh = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self._log_fh.write(f"# command: {self.command_line()}\n")
        self._log_fh.flush()
        kwargs: dict = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            self.proc = subprocess.Popen(
                self.command, stdout=self._log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env=self._env(), cwd=str(Path(self.server_cmd[0]).parent) if not self.is_stub else None,
                **kwargs,
            )
        except OSError as e:
            raise BackendError("backend_not_runnable", f"Could not start {self.server_cmd[0]}: {e}",
                               "Check that the file is the Windows x64 llama-server.exe and not blocked by antivirus "
                               "(right-click > Properties > Unblock).")
        self.started_at = time.perf_counter()
        _LIVE.add(self)
        if IS_WINDOWS:
            try:
                self._job = _windows_kill_on_close_job(self.proc)
            except Exception:
                self._job = None

    def _env(self) -> dict:
        env = scrubbed_env(self.cfg.offline)
        env.pop("LLAMA_API_KEY", None)
        if self.api_key:
            env["LLAMA_API_KEY"] = self.api_key
        return env

    def auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def log_tail(self, n: int = 40) -> list[str]:
        if not self.log_path or not self.log_path.exists():
            return []
        try:
            if self._log_fh:
                self._log_fh.flush()
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return [ln for ln in lines if ln.strip() and not ln.startswith("# command:")][-n:]
        except OSError:
            return []

    def important_log_lines(self, max_lines: int = 8) -> list[str]:
        """The error lines worth showing a user, falling back to the last few lines."""
        tail = self.log_tail()
        errs = [ln for ln in tail if _ERR_LINE.search(ln)]
        seen, out = set(), []
        for ln in errs:
            key = re.sub(r"^[\d.]+\s+", "", ln)       # llama.cpp repeats some errors; drop duplicates
            if key not in seen:
                seen.add(key)
                out.append(ln)
        return out[-max_lines:] if out else tail[-max_lines:]

    def _probe_health(self) -> int | None:
        try:
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
            c.request("GET", "/health")
            r = c.getresponse()
            r.read()
            c.close()
            return r.status
        except (OSError, http.client.HTTPException):
            return None

    def wait_ready(self, timeout: float | None = None, on_tick=None) -> float:
        """Block until /health returns 200. Returns startup seconds."""
        timeout = timeout or self.cfg.startup_timeout_s
        assert self.proc and self.started_at is not None
        deadline = self.started_at + timeout
        while True:
            rc = self.proc.poll()
            if rc is not None:
                raise self.exit_error(rc, during="startup")
            status = self._probe_health()
            if status == 200:
                self.ready_at = time.perf_counter()
                return self.ready_at - self.started_at
            if time.perf_counter() > deadline:
                tail = self.important_log_lines()
                self.stop()
                raise BackendError(
                    "startup_timeout", f"llama-server did not become ready within {timeout:.0f} s",
                    "Large models on slow disks can take a while: raise startup_timeout_s in runtime.toml, "
                    "or try a smaller quantization.",
                    details={"log_tail": tail, "log_file": str(self.log_path)},
                )
            if on_tick:
                on_tick(time.perf_counter() - self.started_at)
            time.sleep(0.05)

    def exit_error(self, rc: int, during: str) -> BackendError:
        code, hint = diagnose(self.log_tail())
        tail = self.important_log_lines()
        return BackendError(code, f"llama-server exited with code {rc} during {during}", hint,
                            details={"log_tail": tail, "log_file": str(self.log_path)})

    # ------------------------------------------------------------- misc --
    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def alive(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    def props(self) -> dict:
        try:
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            c.request("GET", "/props", headers=self.auth_headers())
            r = c.getresponse()
            body = r.read()
            return json.loads(body) if r.status == 200 else {}
        except (OSError, http.client.HTTPException, json.JSONDecodeError):
            return {}

    def stop(self, timeout: float = 10.0) -> None:
        self.stopping = True
        _LIVE.discard(self)
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except OSError:
                pass
        if self._log_fh:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None

    def watch(self, on_exit) -> threading.Thread:
        """Call on_exit(returncode) if the process dies while we did not ask it to."""
        def run():
            assert self.proc
            rc = self.proc.wait()
            if not self.stopping:
                on_exit(rc)
        t = threading.Thread(target=run, daemon=True, name="backend-watch")
        t.start()
        return t


def _q(a: str) -> str:
    return a if re.fullmatch(r"[\w./:=@+-]+", a) else "'" + a.replace("'", "'\\''") + "'"
