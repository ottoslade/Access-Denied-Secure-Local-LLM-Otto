"""Unit tests: GGUF parsing, config, discovery, security helpers."""

import pytest

from docqa_runtime import config as config_mod
from docqa_runtime.discovery import (check_resources, find_server, list_models, model_id, pick_default,
                                     scrubbed_env, select_model)
from docqa_runtime.errors import ConfigError, ResourceError, SecurityError
from docqa_runtime.gateway import validate_chat
from docqa_runtime.gguf import GGUFError, quant_from_filename, read_gguf, write_stub_gguf
from docqa_runtime.offline import is_loopback, require_loopback
from docqa_runtime.stubmodels import create_stub_models


# ------------------------------------------------------------------ GGUF --

def test_gguf_roundtrip(tmp_path):
    p = write_stub_gguf(tmp_path / "m-Q5_K_M.gguf", {
        "general.architecture": "qwen2", "general.name": "Test", "general.file_type": 17,
        "qwen2.context_length": 32768, "docqa.stub": True, "big": 2**40})
    info = read_gguf(p)
    assert info.architecture == "qwen2"
    assert info.quant == "Q5_K_M"
    assert info.context_length == 32768
    assert info.is_stub
    assert info.metadata["big"] == 2**40


@pytest.mark.parametrize("name,q", [
    ("Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", "Q4_K_M"), ("phi-3.5-mini-instruct.Q8_0.gguf", "Q8_0"),
    ("Llama-3.2-3B-Instruct-f16.gguf", "F16"), ("gemma-2-2b-it-IQ4_XS.gguf", "IQ4_XS"), ("model.gguf", None)])
def test_quant_from_filename(name, q):
    assert quant_from_filename(name) == q


def test_gguf_rejects_garbage_and_truncation(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(GGUFError, match="not a GGUF"):
        read_gguf(bad)
    good = write_stub_gguf(tmp_path / "ok.gguf", {"general.architecture": "llama", "general.name": "x" * 50})
    trunc = tmp_path / "trunc.gguf"
    trunc.write_bytes(good.read_bytes()[:40])
    with pytest.raises(GGUFError, match="truncated"):
        read_gguf(trunc)


# ---------------------------------------------------------------- config --

def test_config_defaults_are_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config_mod.load()
    assert cfg.host == "127.0.0.1" and cfg.offline and not cfg.allow_remote and cfg.n_gpu_layers == 0


def test_config_file_and_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runtime.toml").write_text('[server]\nport = 9001\n[backend]\nctx_size = 8192\nmodels_dir = "m"\n')
    cfg = config_mod.load(None, {"ctx_size": 2048, "port": None})
    assert cfg.port == 9001 and cfg.ctx_size == 2048
    assert cfg.models_path == tmp_path / "m"


@pytest.mark.parametrize("toml,code", [
    ("[server]\nprot = 1\n", "config_unknown_key"),
    ("[nope]\nx = 1\n", "config_unknown_section"),
    ("[server]\nport = \"8080\"\n", "config_type"),
    ("[server]\nport = 70000\n", "config_port"),
    ("[server\n", "config_invalid"),
])
def test_config_errors_are_actionable(tmp_path, monkeypatch, toml, code):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runtime.toml").write_text(toml)
    with pytest.raises(ConfigError) as e:
        config_mod.load()
    assert e.value.code == code and e.value.hint


# ------------------------------------------------------------- discovery --

def test_list_models_skips_projectors_and_extra_shards(tmp_path):
    create_stub_models(tmp_path, n_params=1e6, quants=["Q4_K_M"], name="a")
    write_stub_gguf(tmp_path / "mmproj-a-f16.gguf", {"general.architecture": "clip"})
    write_stub_gguf(tmp_path / "big-Q4_K_M-00001-of-00002.gguf", {"general.architecture": "llama"})
    write_stub_gguf(tmp_path / "big-Q4_K_M-00002-of-00002.gguf", {"general.architecture": "llama"})
    (tmp_path / "broken.gguf").write_bytes(b"xx")
    models, problems = list_models(tmp_path)
    assert sorted(m.id for m in models) == ["a-q4_k_m", "big-q4_k_m"]
    assert [p.name for p, _ in problems] == ["broken.gguf"]


def test_model_id_and_default_pick(tmp_path):
    from pathlib import Path
    assert model_id(Path("Qwen2.5-3B-Instruct-Q4_K_M.gguf")) == "qwen2.5-3b-instruct-q4_k_m"
    create_stub_models(tmp_path, n_params=1e6, quants=["F16", "Q8_0", "Q4_K_M"], name="m")
    models, _ = list_models(tmp_path)
    assert pick_default(models, ["Q4_K_M", "Q8_0"]).quant == "Q4_K_M"
    assert pick_default(models, ["Q8_0"]).quant == "Q8_0"


def test_select_model_variants(make_cfg, stub_models):
    assert select_model(make_cfg(model="tiny-stub-q8_0")).quant == "Q8_0"
    assert select_model(make_cfg(model="tiny-stub-Q8_0.gguf")).quant == "Q8_0"
    assert select_model(make_cfg(model=str(stub_models / "tiny-stub-Q8_0.gguf"))).quant == "Q8_0"
    with pytest.raises(ConfigError) as e:
        select_model(make_cfg(model="llama-70b"))
    assert e.value.code == "model_not_found" and "tiny-stub-q4_k_m" in e.value.hint


def test_no_models_hint(make_cfg, tmp_path):
    with pytest.raises(ConfigError) as e:
        select_model(make_cfg(models_dir=str(tmp_path / "empty")))
    assert e.value.code == "no_models" and "stub-models" in e.value.hint


def test_find_server(make_cfg, tmp_path):
    assert "docqa_runtime.stub_server" in " ".join(find_server(make_cfg()))
    with pytest.raises(ConfigError) as e:
        find_server(make_cfg(server_bin=str(tmp_path / "nope" / "llama-server")))
    assert e.value.code == "server_not_found"
    cfg = make_cfg(server_bin="")
    cfg.base_dir = str(tmp_path)
    (tmp_path / "bin").mkdir()
    exe = tmp_path / "bin" / ("llama-server.exe" if __import__("os").name == "nt" else "llama-server")
    exe.write_text("")
    assert find_server(cfg) == [str(exe)]


def test_resource_check_refuses_impossible_model(make_cfg, tmp_path):
    create_stub_models(tmp_path / "huge", n_params=400e9, quants=["F16"], name="huge")
    cfg = make_cfg(models_dir=str(tmp_path / "huge"))
    with pytest.raises(ResourceError) as e:
        check_resources(select_model(cfg), cfg)
    assert "smaller quantization" in e.value.hint


def test_env_scrub(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_HF_REPO", "someone/model")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    monkeypatch.setenv("LLAMA_ARG_MODEL_URL", "https://x")
    env = scrubbed_env(True)
    assert "LLAMA_ARG_HF_REPO" not in env and "HF_TOKEN" not in env and "HTTPS_PROXY" not in env
    assert "LLAMA_ARG_MODEL_URL" not in env
    assert env["LLAMA_ARG_OFFLINE"] == "1"


# -------------------------------------------------------------- security --

def test_loopback_rules():
    assert is_loopback("127.0.0.1") and is_loopback("localhost") and is_loopback("::1") and is_loopback("[::1]")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.5") and not is_loopback("evil.com")
    with pytest.raises(SecurityError):
        require_loopback("0.0.0.0", allow_remote=False)
    require_loopback("0.0.0.0", allow_remote=True)


@pytest.mark.parametrize("body,ok", [
    ({"messages": [{"role": "user", "content": "hi"}]}, True),
    ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}, False),
    ({"messages": [{"role": "user", "content": "hi"}], "temperature": 9}, False),
    ({"messages": [{"role": "user", "content": "hi"}], "stream": "yes"}, False),
    ({"messages": [{"role": "wizard", "content": "hi"}]}, False),
    ({"messages": [{"role": "user", "content": 5}]}, False),
    ({"messages": []}, False),
    ([], False),
])
def test_validate_chat(body, ok):
    assert (validate_chat(body) is None) == ok
