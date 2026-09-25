import os
import socket

import pytest

from docqa_runtime import config as config_mod
from docqa_runtime.stubmodels import create_stub_models

# Make the stub fast for tests (load/generation scaled up 25x). Real memory is still allocated.
os.environ.setdefault("DOCQA_STUB_SPEED", "25")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def stub_models(tmp_path):
    d = tmp_path / "models"
    create_stub_models(d, n_params=40e6, quants=["Q4_K_M", "Q8_0"], name="tiny-stub")
    return d


@pytest.fixture
def make_cfg(tmp_path, stub_models, monkeypatch):
    monkeypatch.chdir(tmp_path)          # no runtime.toml here -> pure defaults

    def _make(**over):
        base = {"server_bin": "stub", "models_dir": str(stub_models), "port": free_port(), "startup_timeout_s": 30.0}
        base.update(over)
        cfg = config_mod.load(None, {k: v for k, v in base.items() if k in {
            "model", "models_dir", "server_bin", "host", "port", "ctx_size", "threads", "n_gpu_layers", "allow_remote"}})
        for k, v in base.items():
            setattr(cfg, k, v)
        cfg.runs_dir = str(tmp_path / "runs")
        return cfg

    return _make
