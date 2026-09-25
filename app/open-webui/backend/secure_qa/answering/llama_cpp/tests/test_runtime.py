"""Integration tests: launcher + gateway + (stub) llama-server, end to end over HTTP."""

import http.client
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from docqa_runtime.client import ClientError, chat, get_json
from docqa_runtime.errors import BackendError, PortInUseError
from docqa_runtime.launcher import Runtime

from .conftest import free_port


def raw(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    h = {"Content-Type": "application/json", **(headers or {})}
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    payload = r.read()
    c.close()
    try:
        return r.status, dict(r.getheaders()), json.loads(payload) if payload else None
    except json.JSONDecodeError:
        return r.status, dict(r.getheaders()), payload.decode()


@pytest.fixture
def runtime(make_cfg):
    rt = Runtime(make_cfg(), quiet=True)
    rt.start()
    yield rt
    rt.stop()


def msg(text="hello"):
    return [{"role": "user", "content": text}]


# ------------------------------------------------------------ happy path --

def test_health_models_and_banner(runtime):
    status, h = get_json(runtime.base_url(), "/health")
    assert status == 200 and h["status"] == "ok" and h["ready"]
    assert h["model"]["quant"] == "Q4_K_M"                     # default pick
    assert h["security"] == {"bind": "127.0.0.1", "loopback_only": True, "offline_mode": True}
    assert h["engine"]["stub"] is True and h["startup_s"] > 0
    status, m = get_json(runtime.base_url(), "/v1/models")
    assert status == 200 and m["object"] == "list" and m["data"][0]["id"] == "tiny-stub-q4_k_m"
    assert "path" not in json.dumps(m)                         # never leak local file paths
    banner = runtime.banner()
    for s in ("/health", "/v1/models", "/v1/chat/completions", "STUB"):
        assert s in banner


def test_chat_non_stream_and_stream(runtime):
    r = chat(runtime.base_url(), msg(), max_tokens=6, stream=False)
    assert r["text"] and r["usage"]["completion_tokens"] == 6 and r["timings"]["predicted_per_second"] > 0
    toks = []
    r = chat(runtime.base_url(), msg(), max_tokens=5, stream=True, on_token=toks.append)
    assert len(toks) == 5 and r["ttft_s"] is not None and r["timings"]["prompt_n"] > 0
    assert r["fingerprint"] == "docqa-stub"


def test_model_aliases_accepted(runtime):
    for name in ("default", "local", "tiny-stub-q4_k_m"):
        st, _, body = raw(runtime.cfg.port, "POST", "/v1/chat/completions", {"model": name, "messages": msg(), "max_tokens": 2})
        assert st == 200, body


# ------------------------------------------------------------------ errors --

@pytest.mark.parametrize("body,status,code", [
    ({"messages": []}, 400, "invalid_request"),
    (b"{not json", 400, "invalid_json"),
    ({"model": "gpt-4o", "messages": msg()}, 404, "model_not_found"),
    ({"messages": msg(), "max_tokens": -5}, 400, "invalid_request"),
])
def test_request_errors_have_code_and_hint(runtime, body, status, code):
    st, _, b = raw(runtime.cfg.port, "POST", "/v1/chat/completions", body)
    assert st == status and b["error"]["code"] == code and b["error"]["hint"] is not None


def test_routing_errors(runtime):
    assert raw(runtime.cfg.port, "GET", "/nope")[0] == 404
    assert raw(runtime.cfg.port, "GET", "/v1/chat/completions")[0] == 405
    assert raw(runtime.cfg.port, "DELETE", "/health")[0] == 405


def test_context_overflow_is_explained(make_cfg):
    rt = Runtime(make_cfg(ctx_size=512), quiet=True)
    rt.start()
    try:
        st, _, b = raw(rt.cfg.port, "POST", "/v1/chat/completions", {"messages": msg("word " * 3000)})
        assert st == 400 and b["error"]["code"] == "context_length_exceeded" and "ctx_size" in b["error"]["hint"]
    finally:
        rt.stop()


# ----------------------------------------------------------------- security --

def test_dns_rebinding_and_foreign_origins_blocked(runtime):
    p = runtime.cfg.port
    st, _, b = raw(p, "GET", "/health", headers={"Host": "attacker.example:8080"})
    assert st == 403 and b["error"]["code"] == "host_not_allowed"
    st, _, b = raw(p, "POST", "/v1/chat/completions", {"messages": msg()}, headers={"Origin": "https://evil.example"})
    assert st == 403 and b["error"]["code"] == "origin_not_allowed"
    st, hdrs, _ = raw(p, "GET", "/health", headers={"Origin": "http://localhost:5173"})
    assert st == 200 and hdrs.get("Access-Control-Allow-Origin") == "http://localhost:5173"
    st, hdrs, _ = raw(p, "OPTIONS", "/v1/chat/completions", headers={"Origin": "http://127.0.0.1:3000"})
    assert st == 204 and "POST" in hdrs.get("Access-Control-Allow-Methods", "")


def test_backend_port_requires_per_launch_key(runtime):
    """The internal llama-server port must not be usable by anything except the gateway."""
    b = runtime.backend
    assert b.api_key
    st, _, body = raw(b.port, "POST", "/v1/chat/completions", {"messages": msg(), "max_tokens": 1})
    assert st == 401
    assert b.api_key not in b.command_line()                    # never on the command line / in logs
    assert b.api_key not in b.log_path.read_text()


def test_backend_bound_to_loopback_only(runtime):
    from docqa_runtime.offline import socket_audit
    audit = socket_audit(runtime.backend.pid)
    if not audit["available"]:
        pytest.skip(audit["reason"])
    assert audit["loopback_only"] and audit["listening"]


def test_prompts_are_not_logged(runtime):
    secret = "CONFIDENTIAL-CONTRACT-7731"
    chat(runtime.base_url(), msg(secret), max_tokens=2)
    logs = runtime.cfg.runs_path / "logs"
    for f in logs.iterdir():
        assert secret not in f.read_text(errors="replace"), f


def test_stream_client_disconnect_does_not_break_runtime(runtime):
    c = http.client.HTTPConnection("127.0.0.1", runtime.cfg.port, timeout=10)
    body = json.dumps({"messages": msg(), "max_tokens": 50, "stream": True})
    c.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    r.readline()
    c.close()                                                  # hang up mid-answer
    time.sleep(0.5)
    assert chat(runtime.base_url(), msg(), max_tokens=2)["text"]


# ---------------------------------------------------------- failure modes --

def test_load_failure_reports_log(make_cfg, monkeypatch):
    monkeypatch.setenv("DOCQA_STUB_FAIL", "load")
    rt = Runtime(make_cfg(), quiet=True)
    with pytest.raises(BackendError) as e:
        rt.start()
    assert e.value.code == "model_load_failed" and e.value.exit_code == 3
    assert any("simulated load failure" in ln for ln in e.value.details["log_tail"])


def test_startup_timeout(make_cfg, monkeypatch):
    monkeypatch.setenv("DOCQA_STUB_FAIL", "hang_load")
    rt = Runtime(make_cfg(startup_timeout_s=1.5), quiet=True)
    with pytest.raises(BackendError) as e:
        rt.start()
    assert e.value.code == "startup_timeout" and "startup_timeout_s" in e.value.hint


def test_health_reports_loading_then_ok(make_cfg, monkeypatch):
    monkeypatch.setenv("DOCQA_STUB_SPEED", "0.5")              # slow load so we can observe it
    rt = Runtime(make_cfg(), quiet=True)
    t = threading.Thread(target=rt.start)
    t.start()
    seen = set()
    deadline = time.time() + 20
    while time.time() < deadline and t.is_alive():
        try:
            st, h = get_json(rt.base_url(), "/health", timeout=2)
            seen.add((st, h["status"]))
            if h["status"] == "loading":
                st2, _, b = raw(rt.cfg.port, "POST", "/v1/chat/completions", {"messages": msg()})
                assert st2 == 503 and b["error"]["code"] == "model_loading"
        except OSError:
            pass
        time.sleep(0.05)
    t.join()
    try:
        assert (503, "loading") in seen
        h = get_json(rt.base_url(), "/health")[1]
        assert h["status"] == "ok", json.dumps(h["error"])
    finally:
        rt.stop()


def test_crash_is_detected_and_reported(make_cfg, monkeypatch):
    monkeypatch.setenv("DOCQA_STUB_FAIL", "crash_after_first")
    rt = Runtime(make_cfg(), quiet=True)
    rt.start()
    try:
        chat(rt.base_url(), msg(), max_tokens=2)
        assert rt.crashed.wait(10)
        st, h = get_json(rt.base_url(), "/health")
        assert st == 503 and h["status"] == "error" and h["error"]["code"]
        with pytest.raises(ClientError) as e:
            chat(rt.base_url(), msg(), max_tokens=2)
        assert e.value.status == 503
    finally:
        rt.stop()


def test_port_in_use(make_cfg, runtime):
    with pytest.raises(PortInUseError) as e:
        Runtime(make_cfg(port=runtime.cfg.port), quiet=True).start()
    assert str(runtime.cfg.port) in e.value.hint


# -------------------------------------------------------------------- CLI --

def test_cli_up_json_ready_file_and_clean_shutdown(stub_models, tmp_path):
    port = free_port()
    ready = tmp_path / "ready.json"
    env = {**os.environ, "DOCQA_STUB_SPEED": "25"}
    p = subprocess.Popen([sys.executable, "-m", "docqa_runtime", "up", "--server-bin", "stub", "--models-dir",
                          str(stub_models), "--port", str(port), "--json", "--ready-file", str(ready)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path, env=env)
    try:
        line = p.stdout.readline()
        info = json.loads(line)
        assert info["status"] == "ok" and info["endpoints"]["chat_completions"].endswith("/v1/chat/completions")
        assert json.loads(ready.read_text())["backend"]["pid"] == info["backend"]["pid"]
        out = subprocess.run([sys.executable, "-m", "docqa_runtime", "chat", "hi", "--url", f"http://127.0.0.1:{port}",
                              "--max-tokens", "3"], capture_output=True, text=True, timeout=60)
        assert out.returncode == 0 and "[stub]" in out.stdout
        backend_pid = info["backend"]["pid"]
    finally:
        p.terminate()
        rc = p.wait(20)
    assert rc == 0
    import psutil
    time.sleep(0.5)
    assert not psutil.pid_exists(backend_pid) or psutil.Process(backend_pid).status() == psutil.STATUS_ZOMBIE


def test_cli_error_exit_codes(tmp_path):
    run = lambda *a: subprocess.run([sys.executable, "-m", "docqa_runtime", *a], capture_output=True, text=True,  # noqa: E731
                                    cwd=tmp_path, timeout=60)
    r = run("up", "--server-bin", "stub", "--models-dir", str(tmp_path / "none"))
    assert r.returncode == 2 and "no_models" in r.stderr
    r = run("up", "--server-bin", "stub", "--host", "0.0.0.0")
    assert r.returncode == 6 and "remote_bind_refused" in r.stderr
    r = run("chat", "hi", "--url", f"http://127.0.0.1:{free_port()}")
    assert r.returncode == 3 and "runtime_not_running" in r.stderr
