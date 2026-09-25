"""Find the llama-server binary and the GGUF models on disk."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config, app_root
from .errors import ConfigError, ResourceError
from .gguf import GGUFError, GGUFInfo, read_gguf

IS_WINDOWS = os.name == "nt"
EXE = ".exe" if IS_WINDOWS else ""
STUB = "stub"
LLAMA_SERVER_RELEASES = "https://github.com/ggml-org/llama.cpp/releases (Windows CPU build: llama-*-bin-win-cpu-x64.zip)"

_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


# ------------------------------------------------------------ server bin --

def stub_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "_stub-server"]
    return [sys.executable, "-m", "docqa_runtime.stub_server"]


def find_server(cfg: Config) -> list[str]:
    """Return the command prefix used to start the backend."""
    if cfg.server_bin.strip().lower() == STUB:
        return stub_command()
    tried: list[str] = []
    if cfg.server_bin:
        p = cfg.resolve(cfg.server_bin)
        if p.is_dir():
            p = p / f"llama-server{EXE}"
        tried.append(str(p))
        if p.is_file():
            return [str(p)]
        raise ConfigError(
            "server_not_found", f"llama-server was not found at {p}",
            f"Fix server_bin in runtime.toml, or remove it to auto-detect. Download: {LLAMA_SERVER_RELEASES}",
        )
    roots = [Path(cfg.base_dir or app_root()), app_root()]
    for root in dict.fromkeys(roots):
        for cand in (root / "bin" / f"llama-server{EXE}", root / f"llama-server{EXE}"):
            tried.append(str(cand))
            if cand.is_file():
                return [str(cand)]
    on_path = shutil.which("llama-server")
    tried.append("PATH")
    if on_path:
        return [on_path]
    raise ConfigError(
        "server_not_found", "Could not find llama-server" + EXE,
        f"Unzip the llama.cpp release into the 'bin' folder next to the program (scripts/windows/fetch_runtime.ps1 does this), "
        f"or set server_bin in runtime.toml. Download: {LLAMA_SERVER_RELEASES}. "
        "To try the launcher without a model, use --server-bin stub.",
        details={"looked_in": "; ".join(tried)},
    )


def server_help(cmd: list[str], timeout: float = 20.0) -> str:
    """Return `llama-server --help` text ('' if it cannot be read)."""
    try:
        out = subprocess.run(cmd + ["--help"], capture_output=True, text=True, timeout=timeout,
                             errors="replace", env=scrubbed_env(),
                             creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0)
        return (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def server_version(cmd: list[str], timeout: float = 20.0) -> str:
    try:
        out = subprocess.run(cmd + ["--version"], capture_output=True, text=True, timeout=timeout,
                             errors="replace", env=scrubbed_env(),
                             creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0)
        text = (out.stdout or "") + (out.stderr or "")
        m = re.search(r"version:\s*([^\r\n]+)", text)
        return m.group(1) if m else text.strip().splitlines()[-1] if text.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# Environment variables that could make llama.cpp reach the network (e.g. -hf
# model downloads) or silently change its behaviour. Removed before spawning.
_SCRUB_PREFIXES = ("LLAMA_ARG_", "HF_", "HUGGING_FACE_", "LLAMA_CACHE", "MODEL_ENDPOINT")


def scrubbed_env(offline: bool = True) -> dict:
    env = dict(os.environ)
    if offline:
        for k in list(env):
            if k.upper().startswith(_SCRUB_PREFIXES):
                env.pop(k)
        env["LLAMA_ARG_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
    # proxies are irrelevant for a loopback-only process; drop them so nothing is tempted to use them
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(k, None)
    return env


# ---------------------------------------------------------------- models --

@dataclass
class ModelEntry:
    id: str
    path: Path
    size_bytes: int
    quant: str | None
    architecture: str | None
    name: str | None
    context_length: int | None
    size_label: str | None
    stub: bool
    stub_ram_bytes: int | None = None

    @property
    def display_size(self) -> str:
        from .metrics import fmt_bytes
        if self.stub and self.stub_ram_bytes:
            return f"{fmt_bytes(self.stub_ram_bytes)} simulated"
        return fmt_bytes(self.size_bytes)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "path": str(self.path), "size_bytes": self.size_bytes, "display_size": self.display_size,
            "weight_bytes": self.stub_ram_bytes if self.stub and self.stub_ram_bytes else self.size_bytes,
            "quant": self.quant,
            "architecture": self.architecture, "name": self.name, "context_length": self.context_length,
            "size_label": self.size_label, "stub": self.stub,
        }

    @classmethod
    def from_path(cls, path: Path) -> "ModelEntry":
        info: GGUFInfo = read_gguf(path)
        return cls(
            id=model_id(path), path=path.resolve(), size_bytes=path.stat().st_size, quant=info.quant,
            architecture=info.architecture, name=info.name, context_length=info.context_length,
            size_label=info.size_label, stub=info.is_stub,
            stub_ram_bytes=info.metadata.get("docqa.stub.ram_bytes") if info.is_stub else None,
        )


def model_id(path: Path) -> str:
    name = _SHARD_RE.sub(".gguf", path.name)
    stem = name[: -len(".gguf")] if name.lower().endswith(".gguf") else path.stem
    return stem.lower()


def list_models(models_dir: Path) -> tuple[list[ModelEntry], list[tuple[Path, str]]]:
    """Return (models, problems). Problems are unreadable files with a reason."""
    models: list[ModelEntry] = []
    problems: list[tuple[Path, str]] = []
    if not models_dir.is_dir():
        return models, problems
    files = sorted(set(models_dir.glob("*.gguf")) | set(models_dir.glob("*/*.gguf")))
    for f in files:
        low = f.name.lower()
        if "mmproj" in low:                         # vision projectors: not a text model
            continue
        m = _SHARD_RE.search(f.name)
        if m and m.group(1) != "00001":             # only the first shard of split models
            continue
        try:
            models.append(ModelEntry.from_path(f))
        except (GGUFError, OSError) as e:
            problems.append((f, str(e)))
    return models, problems


def select_model(cfg: Config) -> ModelEntry:
    want = cfg.model.strip()
    if want:
        direct = cfg.resolve(want)
        if direct.is_file():
            try:
                return ModelEntry.from_path(direct)
            except GGUFError as e:
                raise ConfigError("model_invalid", f"{direct.name} could not be read: {e}",
                                  "Re-download the file; compare its size with the size on the download page.")
    models, problems = list_models(cfg.models_path)
    if want:
        key = want.lower().removesuffix(".gguf")
        for m in models:
            if key in (m.id, m.path.stem.lower()):
                return m
        avail = ", ".join(m.id for m in models) or "(none)"
        raise ConfigError("model_not_found", f"Model '{want}' was not found",
                          f"Available in {cfg.models_path}: {avail}. Run 'docqa-runtime models' to list them.")
    if not models:
        extra = f" {len(problems)} file(s) could not be read: " + "; ".join(f"{p.name}: {r}" for p, r in problems) if problems else ""
        raise ConfigError(
            "no_models", f"No GGUF models found in {cfg.models_path}.{extra}",
            "Copy a .gguf file into that folder (see docs/week1-candidate-models.md for the shortlist), "
            "or run 'docqa-runtime stub-models' to create stub models for a dry run.",
        )
    return pick_default(models, cfg.quant_preference)


def pick_default(models: list[ModelEntry], preference: list[str]) -> ModelEntry:
    rank = {q.upper(): i for i, q in enumerate(preference)}
    return sorted(models, key=lambda m: (rank.get((m.quant or "").upper(), len(rank)), m.size_bytes, m.id))[0]


# ------------------------------------------------------------- resources --

def estimate_ram_bytes(model: ModelEntry, ctx_size: int) -> int:
    """Rough upper estimate of resident memory: weights + KV cache + runtime overhead.

    KV cache is estimated at 0.1 MB/token (fp16, ~3B class, GQA). Deliberately
    conservative - it is only used to fail early with a clear message.
    """
    weights = model.stub_ram_bytes if model.stub and model.stub_ram_bytes else model.size_bytes
    return int(weights + ctx_size * 100 * 1024 + 300 * 1024 * 1024)


def check_resources(model: ModelEntry, cfg: Config) -> list[str]:
    """Raise if the model cannot possibly fit; return warnings if it is tight."""
    import psutil

    need = estimate_ram_bytes(model, cfg.ctx_size)
    vm = psutil.virtual_memory()
    gb = lambda b: f"{b / 1024**3:.1f} GB"  # noqa: E731
    if need > vm.total * 0.9:
        raise ResourceError(
            "insufficient_ram",
            f"{model.id} needs about {gb(need)} but this machine has {gb(vm.total)} RAM",
            "Pick a smaller quantization (e.g. Q4_K_M instead of Q8_0/F16) or a smaller model, or lower ctx_size.",
            details={"model_file": gb(model.size_bytes), "ctx_size": cfg.ctx_size},
        )
    warnings = []
    if need > vm.available:
        warnings.append(
            f"Only {gb(vm.available)} RAM is free and {model.id} needs about {gb(need)}; expect swapping. "
            "Close other applications or pick a smaller quantization."
        )
    return warnings
