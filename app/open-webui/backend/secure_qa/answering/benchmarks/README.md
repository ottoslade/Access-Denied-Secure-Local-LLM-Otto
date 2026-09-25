# Benchmarks

Scaffold only. This feature will hold answer correctness, grounding, citation, refusal, visual-question, model latency, and token-rate evaluation material. The [codebase design](../../../../../../docs/superpowers/specs/2026-09-22-secure-local-document-qa-codebase-design.md) defines the release gates. Retrieval benchmarks belong to Collection Library; whole-process memory checks belong to the desktop shell. No benchmark data or model files are present.

Model runtime benchmarks (startup, peak RAM, tokens/sec, RAG time-to-first-token per quantization) are produced by `docqa-runtime baseline` and `docqa-runtime bench` in [`../llama_cpp/`](../llama_cpp/README.md). Sample outputs are in [`../llama_cpp/docs/sample-results/`](../llama_cpp/docs/sample-results/README.md).
