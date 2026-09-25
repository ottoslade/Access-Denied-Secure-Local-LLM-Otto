"""Tiny stdlib client for the gateway (used by `chat`, `baseline`, `bench`)."""

from __future__ import annotations

import http.client
import json
import time
from urllib.parse import urlsplit


class ClientError(Exception):
    def __init__(self, status: int, body: dict | str):
        self.status = status
        self.body = body
        err = body.get("error", {}) if isinstance(body, dict) else {}
        msg = err.get("message") if err else str(body)[:300]
        hint = err.get("hint", "") if err else ""
        super().__init__(f"HTTP {status}: {msg}" + (f"\n  -> {hint}" if hint else ""))


def _conn(base_url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    u = urlsplit(base_url)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout), u.path.rstrip("/")


def get_json(base_url: str, path: str, timeout: float = 10) -> tuple[int, dict]:
    c, prefix = _conn(base_url, timeout)
    try:
        c.request("GET", prefix + path)
        r = c.getresponse()
        raw = r.read()
        try:
            return r.status, json.loads(raw)
        except json.JSONDecodeError:
            return r.status, {"raw": raw.decode("utf-8", "replace")}
    finally:
        c.close()


def chat(base_url: str, messages: list[dict], *, max_tokens: int = 128, temperature: float = 0.0,
         stream: bool = True, timeout: float = 600, on_token=None, extra: dict | None = None) -> dict:
    """Send a chat request. Returns text, ttft_s, total_s, usage, timings, model."""
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": 42,
            "stream": stream, **(extra or {})}
    if stream:
        body["stream_options"] = {"include_usage": True}
    data = json.dumps(body).encode()
    c, prefix = _conn(base_url, timeout)
    t0 = time.perf_counter()
    try:
        c.request("POST", prefix + "/v1/chat/completions", body=data,
                  headers={"Content-Type": "application/json", "Content-Length": str(len(data))})
        r = c.getresponse()
        if r.status != 200:
            raw = r.read()
            try:
                raise ClientError(r.status, json.loads(raw))
            except json.JSONDecodeError:
                raise ClientError(r.status, raw.decode("utf-8", "replace"))
        if not stream:
            obj = json.loads(r.read())
            total = time.perf_counter() - t0
            return {"text": obj["choices"][0]["message"].get("content") or "", "ttft_s": None, "total_s": total,
                    "usage": obj.get("usage"), "timings": obj.get("timings"), "model": obj.get("model"),
                    "fingerprint": obj.get("system_fingerprint")}
        parts: list[str] = []
        ttft = None
        usage = timings = model = fp = None
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            obj = json.loads(payload)
            model = obj.get("model", model)
            fp = obj.get("system_fingerprint", fp)
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("timings"):
                timings = obj["timings"]
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(piece)
                    if on_token:
                        on_token(piece)
        return {"text": "".join(parts), "ttft_s": ttft, "total_s": time.perf_counter() - t0,
                "usage": usage, "timings": timings, "model": model, "fingerprint": fp}
    finally:
        c.close()
