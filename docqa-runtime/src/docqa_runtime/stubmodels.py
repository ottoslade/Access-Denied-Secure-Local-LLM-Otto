"""Create header-only stub GGUF files so the whole pipeline runs without weights."""

from __future__ import annotations

from pathlib import Path

from .gguf import FILE_TYPE_IDS, write_stub_gguf
from .stub_server import BITS_PER_WEIGHT

DEFAULT_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0", "F16"]


def create_stub_models(out_dir: Path, *, n_params: float = 1.0e9, quants: list[str] | None = None,
                       name: str | None = None) -> list[Path]:
    name = name or f"stub-{n_params / 1e9:g}b-instruct"
    paths = []
    for q in quants or DEFAULT_QUANTS:
        q = q.upper()
        if q not in FILE_TYPE_IDS:
            raise ValueError(f"unknown quantization {q}")
        ram = int(n_params * BITS_PER_WEIGHT.get(q, 5.0) / 8)
        meta = {
            "general.architecture": "llama",
            "general.name": f"{name} (stub)",
            "general.size_label": f"{n_params / 1e9:g}B",
            "general.file_type": FILE_TYPE_IDS[q],
            "llama.context_length": 32768,
            "docqa.stub": True,
            "docqa.stub.n_params": int(n_params),
            "docqa.stub.ram_bytes": ram,
        }
        paths.append(write_stub_gguf(out_dir / f"{name}-{q}.gguf", meta))
    return paths
