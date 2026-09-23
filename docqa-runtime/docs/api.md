# Runtime API contract (E1 → E3 / E4)

**Base URL:** `http://127.0.0.1:8080` (the port is configurable; the host is always loopback).
**Version:** `2026-09-v1`, returned in the `X-DocQA-API-Version` header and in `/health.api_version`.
Anything added within v1 is additive. Breaking changes bump the version.

The runtime is OpenAI-compatible where it matters, so any OpenAI client library works with
`base_url="http://127.0.0.1:8080/v1"` and any `api_key` value (the key is ignored on the public side).

## `GET /health`

Returns **200** only when the model is loaded and ready. Otherwise it returns **503** with the same body,
so the frontend can show *why* it isn't ready.

```json
{
  "status": "ok",                     // starting | loading | ok | error | stopped
  "ready": true,
  "api_version": "2026-09-v1",
  "runtime_version": "0.2.0",
  "engine": {"name": "llama.cpp", "server_version": "6543 (a1b2c3d)", "stub": false, "pid": 4312},
  "model": {"id": "qwen_qwen3-4b-instruct-2507-q4_k_m", "name": "Qwen3 4B Instruct 2507", "quant": "Q4_K_M",
            "architecture": "qwen3", "file_size_bytes": 2497280256, "ctx_size": 4096, "ctx_size_requested": 4096},
  "security": {"bind": "127.0.0.1", "loopback_only": true, "offline_mode": true},
  "uptime_s": 42.1,
  "startup_s": 3.84,
  "requests": {"total": 7, "failed": 0},
  "error": null                       // when status == "error": {"code", "message", "hint", "details"}
}
```

**Frontend guidance:** poll every 1–2 s while `status` is `starting` or `loading` (the response carries `Retry-After: 2`).
Show `error.message` and `error.hint` as-is when `status` is `error`. `engine.stub == true` means no real model
is loaded, so show a "demo mode" badge.

## `GET /v1/models`

```json
{"object": "list", "data": [{
  "id": "qwen_qwen3-4b-instruct-2507-q4_k_m", "object": "model", "created": 1790130000, "owned_by": "docqa-local",
  "ready": true,
  "meta": {"name": "...", "quant": "Q4_K_M", "architecture": "qwen3", "file_size_bytes": 2497280256,
           "context_length_train": 262144, "ctx_size": 4096, "stub": false}
}]}
```

One model is loaded at a time. `GET /v1/models/{id}` returns the single object, or 404. Local file paths are
never exposed.

## `POST /v1/chat/completions`

Standard OpenAI chat request. `model` is optional; `""`, `"default"`, `"local"`, `"docqa"` or the loaded id all
work. Any other value returns 404 `model_not_found`.

```json
{
  "messages": [
    {"role": "system", "content": "Answer only from the passages. Cite as [doc:page]."},
    {"role": "user", "content": "Passages:\n[handbook.pdf:12] ...\n\nQuestion: ..."}
  ],
  "max_tokens": 256,
  "temperature": 0.2,
  "stream": false
}
```

Validated fields: `messages` (non-empty; role ∈ system/user/assistant/tool/developer; content string or parts list),
`max_tokens` 1–32768, `temperature` 0–5, `top_p` 0–1, `stream` boolean. Other llama.cpp sampling fields
(`seed`, `top_k`, `min_p`, `repeat_penalty`, `stop`, `response_format`, `cache_prompt`, …) are passed through.

**Non-streaming response:** the standard OpenAI shape plus llama.cpp's `timings`:

```json
{"id": "chatcmpl-...", "object": "chat.completion", "created": 1790130000, "model": "qwen_...-q4_k_m",
 "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "... [handbook.pdf:13]"}}],
 "usage": {"prompt_tokens": 1312, "completion_tokens": 58, "total_tokens": 1370},
 "timings": {"prompt_n": 1312, "prompt_ms": 5210.4, "prompt_per_second": 251.8,
             "predicted_n": 58, "predicted_ms": 4420.1, "predicted_per_second": 13.1}}
```

**Streaming (`"stream": true`):** Server-Sent Events, `data: {chat.completion.chunk}` lines, ending with
`data: [DONE]`. The final chunk carries `finish_reason` and `timings`. Send `"stream_options": {"include_usage": true}`
to also get a `usage` chunk. This is the mode to use for the chat UI, because the first words appear while the model is
still writing.

## Errors

Every error has the same envelope. `code` is stable and safe to switch on; `hint` is written for end users.

```json
{"error": {"message": "Model 'gpt-4' is not loaded", "type": "not_found_error", "code": "model_not_found",
           "hint": "Omit 'model' or use 'qwen_...-q4_k_m'. Switching models requires restarting the runtime with --model.",
           "status": 404}}
```

| HTTP | code | When |
|---|---|---|
| 400 | `invalid_json`, `invalid_request` | malformed body / failed validation |
| 400 | `context_length_exceeded` | prompt + retrieved chunks longer than `ctx_size` (E3: send fewer or shorter chunks) |
| 403 | `host_not_allowed`, `origin_not_allowed` | request from a non-local web page / DNS-rebinding attempt |
| 404 | `model_not_found`, `not_found` | wrong model id / unknown path |
| 405 | `method_not_allowed` | e.g. GET on `/v1/chat/completions` |
| 411 / 413 / 415 | `length_required`, `request_too_large` (>16 MiB), `unsupported_media_type` | |
| 502 | `backend_disconnected` | model process died mid-answer |
| 503 | `model_loading` | still loading. Retry after `Retry-After` seconds |
| 503 | `backend_unavailable` | model process is not running. `/health.error` says why |
| 504 | `backend_timeout` | no answer within `request_timeout_s` (default 600 s) |

## Browser access (E4)

Requests with an `Origin` header are allowed only from `http://localhost:*`, `http://127.0.0.1:*`, `file://`,
`app://`, `tauri://` or `null` (Electron/Tauri). CORS preflight (`OPTIONS`) is supported for those origins. The
`Host` header must be `localhost` / `127.0.0.1` / `[::1]`. Anything else gets a 403.

## Launcher integration (desktop app)

`docqa-runtime up --json --ready-file <path>` prints a single JSON line and writes the same JSON to `<path>` once the
model is ready:

```json
{"status": "ok", "base_url": "http://127.0.0.1:8080",
 "endpoints": {"health": ".../health", "models": ".../v1/models", "chat_completions": ".../v1/chat/completions"},
 "model": {...}, "startup_s": 3.84, "backend": {"pid": 4312, "port": 53211, "version": "...", "stub": false, ...}}
```

On failure it prints `{"status": "error", "error": {code, message, hint}}` and exits with a non-zero code:

| Exit code | Meaning |
|---|---|
| 0 | clean shutdown |
| 2 | configuration: missing binary / model, bad runtime.toml |
| 3 | llama-server failed to start or crashed |
| 4 | port already in use |
| 5 | not enough RAM for the chosen model |
| 6 | refused for security reasons (non-loopback bind, `--require-offline` with network up) |
