"""Stub llama-server.

A stand-in for llama.cpp's ``llama-server`` that speaks the same CLI flags and
HTTP API (``/health``, ``/v1/models``, ``/props``, ``/v1/chat/completions``
with and without streaming, ``timings`` block included). It lets the launcher,
gateway, tests and benchmark pipeline run end to end with no model weights.

Performance is *simulated* but the memory is *real*: the stub allocates and
touches the number of bytes a model of that size/quantization would occupy,
so the RAM measurement code path is exercised for real. Generation speed is
modelled as memory-bandwidth bound (tokens/s ~ bandwidth / model bytes), which
is how CPU inference behaves on a laptop.

Every response is marked as stub output (``system_fingerprint`` = ``docqa-stub``,
``/props.build_info`` = ``docqa-stub``) so a simulated number can never be
mistaken for a real one.

Fault injection for tests: DOCQA_STUB_FAIL = load | crash_after_first | hang_load,
or a model file with metadata docqa.stub.fail = "load".
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BUILD_INFO = "docqa-stub"
STUB_VERSION = "version: 0 (docqa-stub)"

# Approximate bits per weight for common GGUF quantizations (incl. scales).
BITS_PER_WEIGHT = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0, "Q8_0": 8.5, "Q6_K": 6.56, "Q5_K_M": 5.69, "Q5_K_S": 5.54,
    "Q5_0": 5.5, "Q4_K_M": 4.89, "Q4_K_S": 4.58, "Q4_0": 4.55, "IQ4_XS": 4.25, "Q3_K_M": 3.91,
    "Q3_K_S": 3.5, "Q2_K": 3.35, "IQ3_XS": 3.3, "IQ2_XS": 2.31,
}

HELP_TEXT = """usage: llama-server [options]   (docqa stub)
-m,    --model FNAME
-a,    --alias STRING
-c,    --ctx-size N
-t,    --threads N
-ngl,  --gpu-layers, --n-gpu-layers N
--host HOST
--port PORT
--no-mmap
--mlock
--offline
--no-webui
--no-warmup
--jinja
--log-file FNAME
--api-key KEY
--version
"""

_CANNED = (
    "This is a simulated answer from the DocQA stub runtime. No model weights were loaded, "
    "so the text carries no meaning, but its length, pacing and response format match what the "
    "real llama.cpp server would return for the same request."
).split()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class StubModel:
    def __init__(self, path: Path, alias: str | None, n_ctx: int):
        from .discovery import model_id
        from .gguf import read_gguf

        self.path = path
        info = read_gguf(path)
        if info.metadata.get("docqa.stub.fail") == "load":
            raise RuntimeError("simulated load failure (docqa.stub.fail in model file)")
        self.quant = info.quant or "Q4_K_M"
        self.n_params = int(info.metadata.get("docqa.stub.n_params", 0)) or None
        if self.n_params:
            bpw = BITS_PER_WEIGHT.get(self.quant, 5.0)
            self.weight_bytes = int(self.n_params * bpw / 8)
        else:  # a real GGUF handed to the stub: use its file size
            self.weight_bytes = path.stat().st_size
            self.n_params = int(self.weight_bytes * 8 / BITS_PER_WEIGHT.get(self.quant, 5.0))
        self.alias = alias or model_id(path)
        self.n_ctx = n_ctx
        self.n_ctx_train = info.context_length or 32768
        speed = float(os.environ.get("DOCQA_STUB_SPEED", "1.0"))   # >1 = faster (tests)
        bandwidth = float(os.environ.get("DOCQA_STUB_BANDWIDTH_GBS", "14")) * 1024**3
        self.gen_tps = bandwidth / max(self.weight_bytes, 1) * speed
        self.prompt_tps = self.gen_tps * 9.0
        self.load_s = (0.25 + self.weight_bytes / (1.2 * 1024**3)) / speed
        self._mem: list[bytearray] = []
        self.rng = random.Random(hash(self.alias) & 0xFFFF)

    def load(self) -> None:
        cap = int(float(os.environ.get("DOCQA_STUB_MAX_ALLOC_GB", "2.0")) * 1024**3)
        target = min(self.weight_bytes, cap)
        t_end = time.perf_counter() + self.load_s
        chunk = 64 * 1024 * 1024
        allocated = 0
        while allocated < target:
            n = min(chunk, target - allocated)
            b = bytearray(n)
            b[::4096] = b"\x01" * len(range(0, n, 4096))   # touch every page so it is resident
            self._mem.append(b)
            allocated += n
        # KV cache (fp16, ~0.1 MB/token scaled by model size) - allocated up front like llama.cpp
        kv = min(int(self.n_ctx * 100 * 1024 * min(1.0, self.n_params / 3e9)), cap)
        if kv:
            b = bytearray(kv)
            b[::4096] = b"\x01" * len(range(0, kv, 4096))
            self._mem.append(b)
        rest = t_end - time.perf_counter()
        if rest > 0:
            time.sleep(rest)

    def jitter(self) -> float:
        return 1.0 + self.rng.uniform(-0.03, 0.03)


class State:
    model: StubModel
    ready = threading.Event()
    lock = threading.Lock()
    requests = 0


def count_prompt_tokens(messages) -> int:
    text = ""
    for m in messages or []:
        c = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        text += str(c)
    return max(1, len(text) // 4 + 8 * len(messages or []))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llama-server-stub"

    def log_message(self, fmt, *args):  # keep stderr for the launcher's log
        log("srv  log_server_r: " + (fmt % args))

    # -- helpers --
    def _json(self, status: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _unauthorized(self) -> bool:
        key = os.environ.get("LLAMA_API_KEY")
        if key and self.path not in ("/health", "/v1/health", "/models", "/v1/models"):
            if self.headers.get("Authorization") != f"Bearer {key}":
                self._json(401, {"error": {"code": 401, "message": "Invalid API Key", "type": "authentication_error"}})
                return True
        return False

    def _not_ready(self) -> bool:
        if not State.ready.is_set():
            self._json(503, {"error": {"message": "Loading model", "type": "unavailable_error", "code": 503}})
            return True
        return False

    # -- routes --
    def do_GET(self):
        if self._not_ready() or self._unauthorized():
            return
        m = State.model
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        if self.path in ("/v1/models", "/models"):
            return self._json(200, {
                "object": "list",
                "data": [{
                    "id": m.alias, "object": "model", "created": int(time.time()), "owned_by": "llamacpp",
                    "meta": {"n_params": m.n_params, "size": m.weight_bytes, "n_ctx_train": m.n_ctx_train, "stub": True},
                }],
            })
        if self.path == "/props":
            return self._json(200, {"build_info": BUILD_INFO, "model_alias": m.alias, "model_path": str(m.path),
                                    "default_generation_settings": {"n_ctx": m.n_ctx}, "total_slots": 1})
        self._json(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})

    def do_POST(self):
        if self._not_ready() or self._unauthorized():
            return
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            return self._json(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            return self._json(400, {"error": {"code": 400, "message": f"Invalid JSON: {e}", "type": "invalid_request_error"}})
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._json(400, {"error": {"code": 400, "message": "'messages' is required", "type": "invalid_request_error"}})
        max_tokens = req.get("max_tokens") or req.get("n_predict") or 48
        max_tokens = max(1, min(int(max_tokens), 2048))
        n_prompt = count_prompt_tokens(messages)
        if n_prompt > State.model.n_ctx:
            return self._json(400, {"error": {"code": 400, "type": "exceed_context_size_error",
                                              "message": f"the request exceeds the available context size ({n_prompt} > {State.model.n_ctx} tokens)",
                                              "n_prompt_tokens": n_prompt, "n_ctx": State.model.n_ctx}})
        with State.lock:
            State.requests += 1
            if req.get("stream"):
                self._stream(req, n_prompt, max_tokens)
            else:
                self._complete(req, n_prompt, max_tokens)
        if os.environ.get("DOCQA_STUB_FAIL") == "crash_after_first":
            log("stub: simulated crash (DOCQA_STUB_FAIL=crash_after_first)")
            os._exit(139)

    def _tokens(self, max_tokens: int):
        m = State.model
        for i in range(max_tokens):
            time.sleep(m.jitter() / m.gen_tps)
            yield ("" if i == 0 else " ") + _CANNED[i % len(_CANNED)]

    def _timings(self, n_prompt, t_prompt_ms, n_gen, t_gen_ms) -> dict:
        return {
            "cache_n": 0,
            "prompt_n": n_prompt, "prompt_ms": t_prompt_ms,
            "prompt_per_token_ms": t_prompt_ms / max(n_prompt, 1),
            "prompt_per_second": n_prompt / (t_prompt_ms / 1000) if t_prompt_ms else 0.0,
            "predicted_n": n_gen, "predicted_ms": t_gen_ms,
            "predicted_per_token_ms": t_gen_ms / max(n_gen, 1),
            "predicted_per_second": n_gen / (t_gen_ms / 1000) if t_gen_ms else 0.0,
        }

    def _prompt_phase(self, n_prompt) -> float:
        m = State.model
        t0 = time.perf_counter()
        time.sleep(n_prompt / m.prompt_tps * m.jitter())
        return (time.perf_counter() - t0) * 1000

    def _complete(self, req, n_prompt, max_tokens):
        m = State.model
        t_prompt_ms = self._prompt_phase(n_prompt)
        t0 = time.perf_counter()
        text = "".join(self._tokens(max_tokens))
        t_gen_ms = (time.perf_counter() - t0) * 1000
        self._json(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
            "model": m.alias, "system_fingerprint": BUILD_INFO,
            "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": max_tokens, "total_tokens": n_prompt + max_tokens},
            "timings": self._timings(n_prompt, t_prompt_ms, max_tokens, t_gen_ms),
        })

    def _stream(self, req, n_prompt, max_tokens):
        m = State.model
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": m.alias,
                "system_fingerprint": BUILD_INFO}
        t_prompt_ms = self._prompt_phase(n_prompt)
        send({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"role": "assistant", "content": None}}]})
        t0 = time.perf_counter()
        for tok in self._tokens(max_tokens):
            send({**base, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": tok}}]})
        t_gen_ms = (time.perf_counter() - t0) * 1000
        final = {**base, "choices": [{"index": 0, "finish_reason": "length", "delta": {}}]}
        usage = {"prompt_tokens": n_prompt, "completion_tokens": max_tokens, "total_tokens": n_prompt + max_tokens}
        timings = self._timings(n_prompt, t_prompt_ms, max_tokens, t_gen_ms)
        if (req.get("stream_options") or {}).get("include_usage"):
            send(final)
            send({**base, "choices": [], "usage": usage, "timings": timings})
        else:
            send({**final, "timings": timings})
        send("[DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-m", "--model")
    p.add_argument("-a", "--alias")
    p.add_argument("-c", "--ctx-size", type=int, default=4096)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--version", action="store_true")
    p.add_argument("-h", "--help", action="store_true")
    args, _unknown = p.parse_known_args(argv)
    if args.help:
        print(HELP_TEXT)
        return 0
    if args.version:
        print(STUB_VERSION, file=sys.stderr)
        print("built with docqa stub", file=sys.stderr)
        return 0
    if not args.model:
        log("error: --model is required")
        return 1

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    log(f"main: HTTP server is listening, hostname: {args.host}, port: {args.port}, http threads: 4")
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    log(f"llama_model_loader: loading model from {args.model} (stub)")
    fail = os.environ.get("DOCQA_STUB_FAIL", "")
    try:
        if fail == "load":
            time.sleep(0.3)
            raise RuntimeError("simulated load failure (DOCQA_STUB_FAIL=load)")
        State.model = StubModel(Path(args.model), args.alias, args.ctx_size)
        if fail == "hang_load":
            time.sleep(3600)
        State.model.load()
    except Exception as e:  # mirror llama-server's wording
        log(f"llama_model_load: error loading model: {e}")
        log("main: exiting due to model loading error")
        srv.shutdown()
        return 1
    m = State.model
    log(f"stub: quant={m.quant} n_params={m.n_params} weights={m.weight_bytes / 1024**2:.0f} MiB "
        f"gen={m.gen_tps:.1f} tok/s (simulated)")
    log("main: model loaded")
    log(f"main: server is listening on http://{args.host}:{args.port} - starting the main loop")
    State.ready.set()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
