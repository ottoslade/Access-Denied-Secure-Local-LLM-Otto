"""Secure Local DocQA - E1 LLM runtime.

Starts llama.cpp's ``llama-server`` on loopback, fronts it with a small, stable
gateway (``/health``, ``/v1/models``, ``/v1/chat/completions``) and provides
baseline / quantization benchmarks.
"""

__version__ = "0.2.0"
API_VERSION = "2026-09-v1"
