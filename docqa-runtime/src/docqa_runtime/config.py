"""Configuration: defaults <- runtime.toml <- CLI flags.

Relative paths are resolved against the directory holding runtime.toml (or the
app root when no file is found), so the packaged folder can be moved anywhere
- e.g. copied to a USB stick for the air-gapped machine.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import ConfigError

CONFIG_NAME = "runtime.toml"


def app_root() -> Path:
    """Folder the program lives in (the .exe folder when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


@dataclass
class Config:
    # gateway (the stable API other lanes talk to)
    host: str = "127.0.0.1"
    port: int = 8080
    allow_remote: bool = False          # must be explicitly enabled to bind non-loopback

    # backend (llama-server)
    server_bin: str = ""                # "" = auto-discover
    backend_port: int = 0               # 0 = pick a free loopback port
    models_dir: str = "models"
    model: str = ""                     # path, file name, or model id; "" = auto
    ctx_size: int = 4096
    threads: int = 0                    # 0 = physical cores
    n_gpu_layers: int = 0               # CPU-only by default (reference laptop)
    mmap: bool = True
    mlock: bool = False
    extra_args: list[str] = field(default_factory=list)
    startup_timeout_s: float = 180.0
    request_timeout_s: float = 600.0

    # offline / security
    offline: bool = True                # pass --offline and scrub download env vars

    # preference order when auto-picking a model/quant
    quant_preference: list[str] = field(
        default_factory=lambda: ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q6_K", "Q8_0", "IQ4_XS", "Q4_0", "Q3_K_M", "F16", "BF16"]
    )

    # where logs / run records go
    runs_dir: str = "runs"

    base_dir: str = field(default="", repr=False)   # set by load(); not a user setting
    source: str = field(default="defaults", repr=False)

    # -------------------------------------------------------------- paths --
    def resolve(self, p: str) -> Path:
        path = Path(os.path.expandvars(os.path.expanduser(p)))
        if not path.is_absolute():
            path = Path(self.base_dir or app_root()) / path
        return path

    @property
    def models_path(self) -> Path:
        return self.resolve(self.models_dir)

    @property
    def runs_path(self) -> Path:
        return self.resolve(self.runs_dir)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


_SECTIONS = {
    "server": {"host", "port", "allow_remote"},
    "backend": {"server_bin", "backend_port", "models_dir", "model", "ctx_size", "threads", "n_gpu_layers",
                "mmap", "mlock", "extra_args", "startup_timeout_s", "request_timeout_s", "quant_preference"},
    "security": {"offline"},
    "output": {"runs_dir"},
}


def find_config_file(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise ConfigError("config_not_found", f"Config file not found: {p}",
                              "Check the --config path, or omit it to use runtime.toml next to the program.")
        return p.resolve()
    for cand in (Path.cwd() / CONFIG_NAME, app_root() / CONFIG_NAME):
        if cand.is_file():
            return cand.resolve()
    return None


def load(explicit: str | None = None, overrides: dict | None = None) -> Config:
    cfg = Config()
    path = find_config_file(explicit)
    if path:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError("config_invalid", f"{path.name} is not valid TOML: {e}",
                              "Fix the line mentioned above (strings need quotes, Windows paths need forward slashes or '\\\\').")
        known = {f for s in _SECTIONS.values() for f in s}
        for section, keys in data.items():
            if section not in _SECTIONS:
                raise ConfigError("config_unknown_section", f"Unknown section [{section}] in {path.name}",
                                  f"Valid sections: {', '.join('[' + s + ']' for s in _SECTIONS)}")
            if not isinstance(keys, dict):
                continue
            for k, v in keys.items():
                if k not in _SECTIONS[section]:
                    hint = f"'{k}' belongs in another section." if k in known else f"Valid keys: {', '.join(sorted(_SECTIONS[section]))}"
                    raise ConfigError("config_unknown_key", f"Unknown key '{k}' in [{section}] of {path.name}", hint)
                _set(cfg, k, v, f"[{section}] {k}")
        cfg.base_dir = str(path.parent)
        cfg.source = str(path)
    else:
        cfg.base_dir = str(app_root())
    for k, v in (overrides or {}).items():
        if v is not None:
            _set(cfg, k, v, f"--{k.replace('_', '-')}")
    validate(cfg)
    return cfg


def _set(cfg: Config, key: str, value, where: str) -> None:
    current = getattr(cfg, key)
    expected = type(current)
    if expected is float and isinstance(value, int):
        value = float(value)
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise ConfigError("config_type", f"{where} should be {expected.__name__}, got {value!r}",
                          "Use a number without quotes for numeric settings, true/false for switches.")
    setattr(cfg, key, value)


def validate(cfg: Config) -> None:
    for name in ("port", "backend_port"):
        v = getattr(cfg, name)
        if not (0 <= v <= 65535) or (name == "port" and v == 0):
            raise ConfigError("config_port", f"{name} = {v} is not a valid port", "Use a number between 1024 and 65535.")
    if cfg.ctx_size < 256:
        raise ConfigError("config_ctx", f"ctx_size = {cfg.ctx_size} is too small", "Use at least 2048 for document Q&A.")
    if cfg.threads < 0:
        raise ConfigError("config_threads", "threads cannot be negative", "Use 0 for automatic.")
