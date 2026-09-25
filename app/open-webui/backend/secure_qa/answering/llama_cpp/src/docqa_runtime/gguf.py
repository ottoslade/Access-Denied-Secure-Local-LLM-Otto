"""Minimal GGUF header reader (and a writer for stub model files).

We only need the key/value metadata (architecture, name, quantization, context
length, parameter count). Tensor data is never read, so inspecting a 4 GB model
takes milliseconds. Large arrays (tokenizer vocab) are skipped without being
materialised.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# GGUF value types
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(13)
_SCALAR = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h", T_UINT32: "<I", T_INT32: "<i",
    T_FLOAT32: "<f", T_BOOL: "<?", T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}

# llama.h LLAMA_FTYPE_* -> human name (general.file_type)
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
}
FILE_TYPE_IDS = {v: k for k, v in FILE_TYPES.items()}

# Quant names as they commonly appear in file names, longest first so that
# "Q4_K_M" wins over "Q4_K" / "Q4".
_QUANT_TOKENS = sorted(FILE_TYPE_IDS, key=len, reverse=True)


class GGUFError(ValueError):
    pass


@dataclass
class GGUFInfo:
    path: Path
    version: int
    tensor_count: int
    metadata: dict = field(default_factory=dict)

    @property
    def architecture(self) -> str | None:
        return self.metadata.get("general.architecture")

    @property
    def name(self) -> str | None:
        return self.metadata.get("general.name")

    @property
    def quant(self) -> str | None:
        ft = self.metadata.get("general.file_type")
        if isinstance(ft, int) and ft in FILE_TYPES:
            return FILE_TYPES[ft]
        return quant_from_filename(self.path.name)

    @property
    def context_length(self) -> int | None:
        arch = self.architecture
        return self.metadata.get(f"{arch}.context_length") if arch else None

    @property
    def is_stub(self) -> bool:
        return bool(self.metadata.get("docqa.stub", False))

    @property
    def size_label(self) -> str | None:
        return self.metadata.get("general.size_label")


def quant_from_filename(name: str) -> str | None:
    upper = name.upper().replace("-", "_").replace(".", "_")
    for tok in _QUANT_TOKENS:
        if tok in upper:
            return tok
    return None


class _Reader:
    def __init__(self, f):
        self.f = f

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise GGUFError("unexpected end of file while reading GGUF header (file truncated or still downloading?)")
        return b

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        if n > 1 << 24:
            raise GGUFError(f"implausible string length {n} in GGUF header")
        return self.read(n).decode("utf-8", errors="replace")

    def skip_string(self) -> None:
        n = self.unpack("<Q")
        self.f.seek(n, io.SEEK_CUR)

    def value(self, vtype: int, keep_arrays: bool):
        if vtype in _SCALAR:
            return self.unpack(_SCALAR[vtype])
        if vtype == T_STRING:
            return self.string()
        if vtype == T_ARRAY:
            etype = self.unpack("<I")
            count = self.unpack("<Q")
            if keep_arrays and count <= 64:
                return [self.value(etype, keep_arrays) for _ in range(count)]
            # skip without materialising
            if etype in _SCALAR:
                self.f.seek(struct.calcsize(_SCALAR[etype]) * count, io.SEEK_CUR)
            elif etype == T_STRING:
                for _ in range(count):
                    self.skip_string()
            else:
                for _ in range(count):
                    self.value(etype, False)
            return {"__array__": True, "type": etype, "count": count}
        raise GGUFError(f"unknown GGUF value type {vtype}")


def read_gguf(path: str | Path) -> GGUFInfo:
    path = Path(path)
    with open(path, "rb", buffering=1 << 16) as f:
        r = _Reader(f)
        magic = r.read(4)
        if magic != GGUF_MAGIC:
            raise GGUFError(f"{path.name} is not a GGUF file (magic={magic!r})")
        version = r.unpack("<I")
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = r.unpack("<Q")
        kv_count = r.unpack("<Q")
        meta: dict = {}
        for _ in range(kv_count):
            key = r.string()
            vtype = r.unpack("<I")
            meta[key] = r.value(vtype, keep_arrays=False)
    return GGUFInfo(path=path, version=version, tensor_count=tensor_count, metadata=meta)


# ---------------------------------------------------------------- writer --

def _w_string(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    buf += struct.pack("<Q", len(b)) + b


def write_stub_gguf(path: str | Path, metadata: dict) -> Path:
    """Write a header-only GGUF file (no tensors). Used for stub models only."""
    path = Path(path)
    buf = bytearray()
    buf += GGUF_MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(metadata))
    for key, val in metadata.items():
        _w_string(buf, key)
        if isinstance(val, bool):
            buf += struct.pack("<I", T_BOOL) + struct.pack("<?", val)
        elif isinstance(val, int):
            if 0 <= val < 2**32:
                buf += struct.pack("<I", T_UINT32) + struct.pack("<I", val)
            else:
                buf += struct.pack("<I", T_UINT64) + struct.pack("<Q", val)
        elif isinstance(val, float):
            buf += struct.pack("<I", T_FLOAT32) + struct.pack("<f", val)
        elif isinstance(val, str):
            buf += struct.pack("<I", T_STRING)
            _w_string(buf, val)
        else:
            raise TypeError(f"unsupported stub metadata type for {key}: {type(val)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))
    return path
