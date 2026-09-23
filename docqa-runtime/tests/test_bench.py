"""Baseline and quantization-matrix pipeline (stub backend), plus an optional real llama-server run."""

import json
import os
from pathlib import Path

import pytest

from docqa_runtime.bench import family, models_for_bench, run_baseline, run_matrix
from docqa_runtime.discovery import list_models
from docqa_runtime.gguf import write_stub_gguf


def test_family_strips_quant(stub_models):
    models, _ = list_models(stub_models)
    assert {family(m) for m in models} == {"tiny-stub"}


def test_matrix_end_to_end(make_cfg, tmp_path):
    cfg = make_cfg()
    models = models_for_bench(cfg, None)
    doc, out = run_matrix(cfg, models, runs=1, max_tokens=4, ram_budget_gb=6, out_dir=tmp_path / "bench", log=lambda *_: None)
    assert all(r["ok"] for r in doc["results"]), doc["analysis"]["failed"]
    for f in ("results.json", "results.csv", "report.md"):
        assert (out / f).exists()
    report = (out / "report.md").read_text()
    assert "SIMULATED" in report and "Q4_K_M vs Q8_0" in report
    q4, q8 = sorted(doc["results"], key=lambda r: r["quant"])
    assert q4["peak_rss_bytes"] < q8["peak_rss_bytes"]             # real memory measurement sees the difference
    assert doc["analysis"]["recommended"]["quant"] in ("Q4_K_M", "Q8_0")
    rag = q4["runs"]["rag"][0]
    assert rag["prompt_n"] > 800                                   # ~1.3k-token RAG prompt, cold (not cached)


def test_matrix_keeps_going_when_a_model_fails(make_cfg, tmp_path):
    cfg = make_cfg()
    write_stub_gguf(Path(cfg.models_dir) / "broken-Q4_K_M.gguf",
                    {"general.architecture": "llama", "docqa.stub": True, "docqa.stub.fail": "load"})
    models = models_for_bench(cfg, ["broken", "tiny-stub-q8_0"])
    doc, out = run_matrix(cfg, models, runs=1, max_tokens=2, ram_budget_gb=6, out_dir=tmp_path / "b2", log=lambda *_: None)
    assert [r["ok"] for r in doc["results"]] == [False, True]
    assert doc["analysis"]["failed"][0]["error"]["code"] == "model_load_failed"
    assert "FAILED: model_load_failed" in (out / "report.md").read_text()


def test_baseline_record(make_cfg, tmp_path, capsys):
    rec, out = run_baseline(make_cfg(), require_offline=False, out_dir=tmp_path / "base", log=lambda *_: None)
    m = rec["metrics"]
    assert m["startup_s"] > 0 and m["peak_rss_bytes"] > 0 and m["gen_tps"] > 0
    assert rec["prompt"].startswith("In two sentences")
    assert "--offline" in rec["backend"]["command"] and "--host 127.0.0.1" in rec["backend"]["command"]
    assert "outbound_reachable" in rec["network"]["before"]
    md = (out / "baseline.md").read_text()
    assert "Exact command" in md and "Peak RAM" in md
    assert json.loads((out / "baseline.json").read_text())["kind"] == "baseline"


# ------------------------------------------------ real llama.cpp (optional) --

REAL_BIN = os.environ.get("DOCQA_TEST_LLAMA_SERVER")
REAL_MODELS = os.environ.get("DOCQA_TEST_MODELS_DIR")


@pytest.mark.skipif(not (REAL_BIN and REAL_MODELS), reason="set DOCQA_TEST_LLAMA_SERVER and DOCQA_TEST_MODELS_DIR")
def test_real_llama_server_matrix(make_cfg, tmp_path):
    cfg = make_cfg(server_bin=REAL_BIN, models_dir=REAL_MODELS)
    models = models_for_bench(cfg, None)[:2]
    doc, _ = run_matrix(cfg, models, runs=1, max_tokens=8, ram_budget_gb=6, out_dir=tmp_path / "real", log=lambda *_: None)
    assert all(r["ok"] for r in doc["results"]), doc["analysis"]["failed"]
    assert not doc["stub"]
    for r in doc["results"]:
        assert r["summary"]["short_gen_tps"] and r["summary"]["rag_prompt_tps"]
        assert "--offline" in r["command"]
