"""`docqa-runtime up`: the one-command launcher."""

from __future__ import annotations

import json
import signal
import sys
import threading
import time
from pathlib import Path

from .backend import BackendProcess
from .config import Config
from .discovery import check_resources, find_server, select_model
from .errors import BackendError, RuntimeFailure
from .gateway import RuntimeState, make_server
from .offline import ensure_port_free, require_loopback


def say(msg: str = "") -> None:
    print(msg, flush=True)


class Runtime:
    """Gateway + backend, started and stopped together."""

    def __init__(self, cfg: Config, quiet: bool = False):
        self.cfg = cfg
        self.quiet = quiet
        self.state: RuntimeState | None = None
        self.backend: BackendProcess | None = None
        self.httpd = None
        self._http_thread: threading.Thread | None = None
        self.crashed = threading.Event()
        self.warnings: list[str] = []

    def _say(self, msg: str = "") -> None:
        if not self.quiet:
            say(msg)

    def base_url(self) -> str:
        host = self.cfg.host if ":" not in self.cfg.host else f"[{self.cfg.host}]"
        return f"http://{host}:{self.cfg.port}"

    def start(self) -> None:
        cfg = self.cfg
        # 1. fail fast on anything we can check before spawning processes
        require_loopback(cfg.host, cfg.allow_remote)
        server_cmd = find_server(cfg)
        model = select_model(cfg)
        self.warnings = check_resources(model, cfg)
        ensure_port_free(cfg.host, cfg.port, "gateway", "--port")

        # 2. gateway first, so /health answers "loading" while the model loads
        self.state = RuntimeState(cfg, model)
        log_dir = cfg.runs_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        access_log_path = log_dir / "gateway-access.log"

        def access_log(line: str, _p=access_log_path):
            try:
                with open(_p, "a", encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
            except OSError:
                pass

        self.httpd = make_server(self.state, access_log)
        self._http_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="gateway")
        self._http_thread.start()

        # 3. backend
        self._say(f"Starting llama.cpp with {model.id} ({model.quant or '?'}, {model.display_size})...")
        for w in self.warnings:
            self._say(f"WARNING: {w}")
        self.backend = BackendProcess(cfg, model, server_cmd, log_dir)
        self.state.backend = self.backend
        self.state.set("loading")
        try:
            self.backend.start()
            startup = self.backend.wait_ready()
        except RuntimeFailure as e:
            self.state.set("error", e.to_dict())
            self.stop()
            raise
        self.state.startup_s = startup
        n_ctx = (self.backend.props().get("default_generation_settings") or {}).get("n_ctx")
        if isinstance(n_ctx, int) and n_ctx > 0:
            self.state.ctx_effective = n_ctx
        self.state.set("ok")
        self.backend.watch(self._on_backend_exit)

    def _on_backend_exit(self, rc: int) -> None:
        err = self.backend.exit_error(rc, during="operation") if self.backend else BackendError("backend_exited", "stopped", "")
        if self.state:
            self.state.set("error", err.to_dict())
        self._say("\n" + err.render())
        self.crashed.set()

    def banner(self, interactive: bool = False) -> str:
        b, m, cfg = self.backend, self.state.model, self.cfg
        url = self.base_url()
        stub = "  [STUB BACKEND - simulated model, no real inference]" if (b.is_stub or m.stub) else ""
        ctx = self.state.ctx_effective or cfg.ctx_size
        ctx_note = f" (requested {cfg.ctx_size}; capped by the model)" if ctx != cfg.ctx_size else ""
        lines = [
            "",
            f"DocQA runtime ready in {self.state.startup_s:.2f} s{stub}",
            f"  Model     {m.id}  ({m.quant or '?'}, {m.display_size}, ctx {ctx}{ctx_note})",
            f"  Health    GET  {url}/health",
            f"  Models    GET  {url}/v1/models",
            f"  Chat      POST {url}/v1/chat/completions",
            f"  Engine    llama-server {b.version}  (pid {b.pid}, internal port 127.0.0.1:{b.port})",
            f"  Offline   {'on' if cfg.offline else 'OFF'}; bound to {cfg.host} ({'this computer only' if not cfg.allow_remote else 'REMOTE ACCESS ENABLED'})",
            f"  Log       {b.log_path}",
            "",
        ]
        if not interactive:
            lines += [
                'Try:  docqa-runtime chat "Hello, are you running locally?"' + (f" --url {url}" if cfg.port != 8080 else ""),
                "Stop: Ctrl+C",
                "",
            ]
        return "\n".join(lines)

    def ready_info(self) -> dict:
        b = self.backend
        return {
            "status": "ok", "base_url": self.base_url(),
            "endpoints": {"health": f"{self.base_url()}/health", "models": f"{self.base_url()}/v1/models",
                          "chat_completions": f"{self.base_url()}/v1/chat/completions"},
            "model": self.state.model.to_dict(), "startup_s": self.state.startup_s,
            "backend": {"pid": b.pid, "port": b.port, "version": b.version, "stub": b.is_stub,
                        "command": b.command, "log": str(b.log_path)},
        }

    def stop(self) -> None:
        if self.backend:
            self.backend.stop()
        if self.state and self.state.status != "error":
            self.state.set("stopped")
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


def run_up(cfg: Config, *, json_output: bool = False, ready_file: str | None = None, after_ready=None) -> int:
    """Start the runtime and block until Ctrl+C or a backend crash.

    after_ready(base_url), if given, runs in the foreground once the runtime is up (e.g. an interactive chat);
    the runtime shuts down when it returns.
    """
    rt = Runtime(cfg, quiet=json_output)
    stop_evt = threading.Event()

    def handle_sig(signum, frame):
        stop_evt.set()

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    # until the runtime is up, termination requests abort startup cleanly (backend is stopped, not orphaned)
    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupt)
    try:
        rt.start()                          # Ctrl+C while loading -> KeyboardInterrupt -> clean stop below
    except RuntimeFailure as e:
        if json_output:
            say(json.dumps({"status": "error", "error": e.to_dict()}))
        else:
            print(e.render(), file=sys.stderr, flush=True)
        return e.exit_code
    except KeyboardInterrupt:
        rt.stop()
        return 130

    signal.signal(signal.SIGINT, handle_sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_sig)
    if hasattr(signal, "SIGBREAK"):          # Ctrl+Break / console close on Windows
        signal.signal(signal.SIGBREAK, handle_sig)

    info = rt.ready_info()
    if ready_file:
        Path(ready_file).write_text(json.dumps(info, indent=2), encoding="utf-8")
    if json_output:
        say(json.dumps(info))
    else:
        say(rt.banner(interactive=after_ready is not None))

    if after_ready is not None:
        signal.signal(signal.SIGINT, signal.default_int_handler)   # Ctrl+C reaches the chat prompt
        try:
            after_ready(rt.base_url())
        except KeyboardInterrupt:
            pass
        stop_evt.set()
    while not stop_evt.is_set() and not rt.crashed.is_set():
        stop_evt.wait(0.5)
    crashed = rt.crashed.is_set()
    if not json_output:
        say("Stopping..." if not crashed else "Backend stopped unexpectedly; shutting down.")
    rt.stop()
    return BackendError.exit_code if crashed else 0
