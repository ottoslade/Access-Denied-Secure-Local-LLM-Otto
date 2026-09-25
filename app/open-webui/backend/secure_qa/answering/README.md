# Question Answering

Scaffold only. This module owns retrieval decisions, deterministic visual selection, inference, evidence classification, and citation validation. Its llama.cpp adapter will stay private to this feature. See the [interface](../../../../../docs/superpowers/specs/2026-09-22-secure-local-document-qa-codebase-design.md#42-question-answering).

The llama.cpp adapter lives in [`llama_cpp/`](llama_cpp/README.md): a one-command launcher for `llama-server` behind a loopback-only, OpenAI-compatible gateway (`/health`, `/v1/models`, `/v1/chat/completions`), plus the baseline and quantization benchmarks. It remains a self-contained Python project (`pyproject.toml`, tests, Windows packaging scripts) until the Open WebUI backend is vendored. Its API contract is [`llama_cpp/docs/api.md`](llama_cpp/docs/api.md).
