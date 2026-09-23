"""The stable local API other lanes code against.

    GET  /health                -> runtime + model state (200 ready, 503 otherwise)
    GET  /v1/models             -> OpenAI-style model list (the loaded model)
    GET  /v1/models/{id}
    POST /v1/chat/completions   -> OpenAI-compatible; streaming supported

Why a gateway instead of exposing llama-server directly:
  * the contract stays fixed when llama.cpp changes (it moves fast);
  * /health reports "loading" / "error" with a reason, not just an HTTP code;
  * every error has a code + hint the frontend can show as-is;
  * Host/Origin checks block DNS-rebinding and random web pages from using the model;
  * request bodies (document text!) are never written to logs.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from . import API_VERSION, __version__
from .offline import is_loopback

if TYPE_CHECKING:
    from .backend import BackendProcess
    from .config import Config
    from .discovery import ModelEntry

MAX_BODY = 16 * 1024 * 1024            # 16 MiB: plenty for retrieved chunks, blocks accidental huge uploads
VALID_ROLES = {"system", "user", "assistant", "tool", "developer"}
MODEL_ALIASES = {"", "default", "local", "docqa"}
ENDPOINTS = ["GET /health", "GET /v1/models", "GET /v1/models/{id}", "POST /v1/chat/completions"]


class RuntimeState:
    """Shared, thread-safe view of the runtime used by /health."""

    def __init__(self, cfg: "Config", model: "ModelEntry"):
        self.cfg = cfg
        self.model = model
        self.backend: "BackendProcess | None" = None
        self.status = "starting"          # starting -> loading -> ok | error -> stopped
        self.error: dict | None = None
        self.startup_s: float | None = None
        self.ctx_effective: int | None = None   # llama.cpp may cap ctx at the model's training length
        self.started_wall = time.time()
        self.lock = threading.Lock()
        self.requests_total = 0
        self.requests_failed = 0

    def set(self, status: str, error: dict | None = None) -> None:
        with self.lock:
            self.status = status
            self.error = error

    def health(self) -> dict:
        b = self.backend
        m = self.model
        with self.lock:
            return {
                "status": self.status,
                "ready": self.status == "ok",
                "api_version": API_VERSION,
                "runtime_version": __version__,
                "engine": {
                    "name": "llama.cpp",
                    "server_version": b.version if b else None,
                    "stub": bool(b and b.is_stub) or m.stub,
                    "pid": b.pid if b else None,
                },
                "model": {
                    "id": m.id, "name": m.name, "quant": m.quant, "architecture": m.architecture,
                    "file_size_bytes": m.size_bytes,
                    "ctx_size": self.ctx_effective or self.cfg.ctx_size, "ctx_size_requested": self.cfg.ctx_size,
                },
                "security": {"bind": self.cfg.host, "loopback_only": is_loopback(self.cfg.host),
                             "offline_mode": self.cfg.offline},
                "uptime_s": round(time.time() - self.started_wall, 1),
                "startup_s": round(self.startup_s, 3) if self.startup_s is not None else None,
                "requests": {"total": self.requests_total, "failed": self.requests_failed},
                "error": self.error,
            }


def error_body(status: int, code: str, message: str, hint: str = "", etype: str | None = None) -> dict:
    etype = etype or {400: "invalid_request_error", 404: "not_found_error", 405: "invalid_request_error",
                      413: "invalid_request_error", 403: "permission_error", 503: "unavailable_error"}.get(status, "server_error")
    return {"error": {"message": message, "type": etype, "code": code, "hint": hint, "status": status}}


# Backend (llama-server) error type -> hint for the user
_BACKEND_HINTS = {
    "exceed_context_size_error": ("context_length_exceeded",
                                  "The prompt plus retrieved passages is longer than the model context. Send fewer/shorter "
                                  "chunks, or raise ctx_size in runtime.toml (uses more RAM)."),
    "invalid_request_error": ("invalid_request", "Check the request body against docs/api.md."),
    "unavailable_error": ("model_loading", "The model is still loading. Poll GET /health until status is 'ok'."),
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"docqa-runtime/{__version__}"
    state: RuntimeState            # injected by make_server
    access_log = None              # callable(str) or None

    # ------------------------------------------------------------ helpers --
    def log_message(self, fmt, *args):   # silence default stderr logging (we log ourselves, without bodies)
        pass

    def _log(self, status: int, t0: float) -> None:
        if self.access_log:
            self.access_log(f"{self.command} {urlsplit(self.path).path} -> {status} ({(time.perf_counter() - t0) * 1000:.0f} ms)")

    def _cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin and self._origin_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_json(self, status: int, obj, extra_headers: dict | None = None) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-DocQA-API-Version", API_VERSION)
        self.send_header("Cache-Control", "no-store")
        self._cors_headers()
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status: int, code: str, message: str, hint: str = "", headers: dict | None = None, etype=None) -> int:
        if status >= 400 and self.command == "POST":
            with self.state.lock:
                self.state.requests_failed += 1
        self._send_json(status, error_body(status, code, message, hint, etype), headers)
        return status

    # --------------------------------------------------- request guards --
    def _origin_allowed(self, origin: str) -> bool:
        if self.state.cfg.allow_remote:
            return True
        if origin in ("null",) or origin.startswith(("file://", "app://", "tauri://")):
            return True
        host = urlsplit(origin).hostname or ""
        return is_loopback(host)

    def _guard(self) -> int | None:
        """DNS-rebinding + cross-site protection. Returns a status if rejected."""
        if self.state.cfg.allow_remote:
            return None
        host_hdr = (self.headers.get("Host") or "").strip()
        hostname = urlsplit("//" + host_hdr).hostname or ""
        if host_hdr and not is_loopback(hostname):
            return self._error(403, "host_not_allowed", f"Host '{host_hdr}' is not allowed",
                               "Call the runtime as http://127.0.0.1:<port> or http://localhost:<port>.")
        origin = self.headers.get("Origin")
        if origin and not self._origin_allowed(origin):
            return self._error(403, "origin_not_allowed", f"Origin '{origin}' is not allowed",
                               "Only pages served from this computer (localhost) or the desktop app may call the runtime.")
        return None

    # ------------------------------------------------------------ routes --
    def do_OPTIONS(self):
        t0 = time.perf_counter()
        if self._guard():
            return
        self.send_response(204)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()
        self._log(204, t0)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        t0 = time.perf_counter()
        status = self._guard() or self._route_get()
        self._log(status, t0)

    def do_POST(self):
        t0 = time.perf_counter()
        status = self._guard() or self._route_post()
        self._log(status, t0)

    def do_PUT(self):
        self._error(405, "method_not_allowed", f"{self.command} is not supported", "Supported: " + ", ".join(ENDPOINTS))

    do_DELETE = do_PATCH = do_PUT

    def _route_get(self) -> int:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        st = self.state
        if path == "/health":
            h = st.health()
            code = 200 if h["ready"] else 503
            self._send_json(code, h, {"Retry-After": "2"} if h["status"] in ("starting", "loading") else None)
            return code
        if path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [self._model_obj()]})
            return 200
        if path.startswith("/v1/models/"):
            mid = path[len("/v1/models/"):]
            if mid.lower() in {st.model.id, *MODEL_ALIASES - {""}}:
                self._send_json(200, self._model_obj())
                return 200
            return self._error(404, "model_not_found", f"Model '{mid}' is not loaded", f"Loaded model: {st.model.id}")
        if path == "/":
            self._send_json(200, {"name": "docqa-runtime", "version": __version__, "api_version": API_VERSION,
                                  "endpoints": ENDPOINTS})
            return 200
        if path == "/v1/chat/completions":
            return self._error(405, "method_not_allowed", "Use POST for /v1/chat/completions", "")
        return self._error(404, "not_found", f"No endpoint {path}", "Available: " + ", ".join(ENDPOINTS))

    def _model_obj(self) -> dict:
        st = self.state
        m = st.model
        return {
            "id": m.id, "object": "model", "created": int(st.started_wall), "owned_by": "docqa-local",
            "ready": st.status == "ok",
            "meta": {"name": m.name, "quant": m.quant, "architecture": m.architecture, "file_size_bytes": m.size_bytes,
                     "context_length_train": m.context_length, "ctx_size": st.ctx_effective or st.cfg.ctx_size,
                     "stub": m.stub},
        }

    def _route_post(self) -> int:
        path = urlsplit(self.path).path.rstrip("/")
        if path in ("/health", "/v1/models"):
            return self._error(405, "method_not_allowed", f"Use GET for {path}", "")
        if path != "/v1/chat/completions":
            return self._error(404, "not_found", f"No endpoint {path}", "Available: " + ", ".join(ENDPOINTS))
        st = self.state
        with st.lock:
            st.requests_total += 1

        # ---- read + validate body
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked" or length < 0:
            return self._error(411, "length_required", "Content-Length header is required", "Send the JSON body with a Content-Length.")
        if length > MAX_BODY:
            return self._error(413, "request_too_large", f"Request body is {length} bytes (limit {MAX_BODY})",
                               "Send fewer or shorter document chunks.")
        raw = self.rfile.read(length) if length else b""
        ctype = (self.headers.get("Content-Type") or "application/json").split(";")[0].strip().lower()
        if ctype not in ("application/json", "text/plain", ""):
            return self._error(415, "unsupported_media_type", f"Content-Type {ctype} is not supported", "Send application/json.")
        try:
            req = json.loads(raw or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return self._error(400, "invalid_json", f"Body is not valid JSON: {e}", "Send a JSON object like {\"messages\": [...]}.")
        problem = validate_chat(req)
        if problem:
            return self._error(400, "invalid_request", problem, "See docs/api.md for the request format.")
        want = str(req.get("model") or "").lower()
        if want not in MODEL_ALIASES and want != st.model.id:
            return self._error(404, "model_not_found", f"Model '{req.get('model')}' is not loaded",
                               f"Omit 'model' or use '{st.model.id}'. Switching models requires restarting the runtime with --model.")
        req["model"] = st.model.id

        # ---- backend state
        if st.status != "ok" or not st.backend:
            if st.status in ("starting", "loading"):
                return self._error(503, "model_loading", "The model is still loading",
                                   "Retry in a few seconds; poll GET /health until status is 'ok'.", {"Retry-After": "2"})
            err = st.error or {}
            return self._error(503, "backend_unavailable", "The model runtime is not running: " + err.get("message", st.status),
                               err.get("hint", "Restart docqa-runtime."))
        return self._forward(req)

    def _forward(self, req: dict) -> int:
        st = self.state
        body = json.dumps(req).encode("utf-8")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", st.backend.port, timeout=st.cfg.request_timeout_s)
            conn.request("POST", "/v1/chat/completions", body=body,
                         headers={"Content-Type": "application/json", "Content-Length": str(len(body)),
                                  **st.backend.auth_headers()})
            resp = conn.getresponse()
        except TimeoutError:
            return self._error(504, "backend_timeout", f"The model did not answer within {st.cfg.request_timeout_s:.0f} s",
                               "Lower max_tokens, send less context, or raise request_timeout_s in runtime.toml.")
        except OSError as e:
            return self._error(503, "backend_unavailable", f"Could not reach llama-server: {e}",
                               "The model process may have crashed - check GET /health and the runtime console.")
        try:
            if resp.status != 200:
                raw = resp.read()
                return self._backend_error(resp.status, raw)
            if req.get("stream"):
                return self._pipe_stream(resp)
            raw = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("X-DocQA-API-Version", API_VERSION)
            self.send_header("Cache-Control", "no-store")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(raw)
            return 200
        except (OSError, http.client.HTTPException) as e:
            # client went away mid-stream, or backend died mid-answer
            if not self.wfile.closed:
                try:
                    return self._error(502, "backend_disconnected", f"The model stopped mid-answer: {e}",
                                       "Check GET /health; the runtime console shows why.")
                except OSError:
                    pass
            return 499
        finally:
            conn.close()

    def _backend_error(self, status: int, raw: bytes) -> int:
        try:
            err = json.loads(raw).get("error", {})
        except (json.JSONDecodeError, AttributeError):
            err = {}
        btype = err.get("type", "")
        msg = err.get("message") or raw.decode("utf-8", "replace")[:500] or f"llama-server returned HTTP {status}"
        code, hint = _BACKEND_HINTS.get(btype, ("backend_error", "See the runtime console / log for details."))
        return self._error(status if 400 <= status < 600 else 502, code, msg, hint, etype=btype or None)

    def _pipe_stream(self, resp) -> int:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-DocQA-API-Version", API_VERSION)
        self._cors_headers()
        self.end_headers()
        while True:
            line = resp.readline()        # SSE is line-oriented; forward each line as it arrives
            if not line:
                break
            self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        return 200


def validate_chat(req) -> str | None:
    if not isinstance(req, dict):
        return "Body must be a JSON object"
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return "'messages' must be a non-empty list"
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            return f"messages[{i}] must be an object"
        if m.get("role") not in VALID_ROLES:
            return f"messages[{i}].role must be one of {sorted(VALID_ROLES)}"
        c = m.get("content")
        if not (isinstance(c, str) or isinstance(c, list) or (c is None and m.get("role") == "assistant")):
            return f"messages[{i}].content must be a string"
    for key, lo, hi in (("max_tokens", 1, 32768), ("n_predict", -1, 32768)):
        if key in req and req[key] is not None:
            v = req[key]
            if not isinstance(v, int) or isinstance(v, bool) or not (lo <= v <= hi):
                return f"'{key}' must be an integer between {lo} and {hi}"
    for key, lo, hi in (("temperature", 0.0, 5.0), ("top_p", 0.0, 1.0)):
        if key in req and req[key] is not None:
            v = req[key]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo <= v <= hi):
                return f"'{key}' must be a number between {lo} and {hi}"
    if "stream" in req and not isinstance(req["stream"], bool):
        return "'stream' must be true or false"
    return None


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False   # never silently share a port on Windows

    def handle_error(self, request, client_address):
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, TimeoutError)):
            return                  # client hung up (normal after streaming / on page reload); not an error
        super().handle_error(request, client_address)


def make_server(state: RuntimeState, access_log=None) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"state": state, "access_log": staticmethod(access_log) if access_log else None})
    return _Server((state.cfg.host, state.cfg.port), handler)
