# Access-Denied-Secure-Local-LLM
Develop and evaluate a locally deployed AI platform for organizations that cannot use public AI services because of sensitive, proprietary, export-controlled, security-related, privacy-protected, or otherwise restricted information.

This repository is currently a **codebase scaffold**. It contains architecture and ownership documentation, without application implementation or vendored dependencies.

- [Codebase design](docs/superpowers/specs/2026-09-22-secure-local-document-qa-codebase-design.md)
- [ADR 0001: Offline Open WebUI in Tauri](docs/architecture/decisions/0001-open-webui-offline.md)
- [ADR 0002: Feature-first source layout](docs/architecture/decisions/0002-feature-first-layout.md)
- [Application layout](app/README.md)
- [Desktop shell](app/desktop/README.md)
- [Open WebUI frontend and backend](app/open-webui/README.md)
- [Release area](app/release/README.md)
- [E1 LLM runtime report](docs/reports/E1-runtime-report.md) (llama.cpp adapter in [`answering/llama_cpp/`](app/open-webui/backend/secure_qa/answering/llama_cpp/README.md))
