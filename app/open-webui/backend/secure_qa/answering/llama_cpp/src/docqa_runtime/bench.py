"""Week 1 baseline run and Week 2 quantization matrix."""

from __future__ import annotations

import csv
import dataclasses
import json
import re
import statistics
import time
from pathlib import Path

from . import __version__
from .client import chat
from .config import Config
from .discovery import ModelEntry, list_models
from .errors import ConfigError, RuntimeFailure, SecurityError
from .gguf import FILE_TYPE_IDS
from .launcher import Runtime
from .metrics import PeakSampler, current_rss, fmt_bytes, system_info
from .offline import free_loopback_port, outbound_network, socket_audit
from .prompts import PROMPT_SET_VERSION, baseline_messages, rag_messages

SIM_BANNER = ("> **SIMULATED RESULTS - stub backend.** These numbers come from the stub llama-server, not a real model. "
              "They prove the pipeline works end to end; re-run on the reference laptop with real GGUF files for real numbers.")

# llama-server keeps the previous prompt in its KV cache and would skip re-reading identical passages on runs 2..N,
# making prompt processing look instant. Benchmarks must measure a cold read every time.
NO_CACHE = {"cache_prompt": False}

# A model is only "usable" for the recommendation if it is at least this fast (defaults; override with
# --min-gen-tps / --max-ttft-s). People read at roughly 5-8 tokens/s; waiting >15 s for the first word feels broken.
MIN_GEN_TPS = 8.0
MAX_RAG_TTFT_S = 15.0

# rough precision rank for choosing the "reference" quant within a model family
_PRECISION = {"F32": 32, "F16": 16, "BF16": 16, "Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7, "Q5_K_S": 5.5, "Q5_0": 5.5,
              "Q4_K_M": 4.9, "Q4_K_S": 4.6, "Q4_0": 4.5, "IQ4_XS": 4.3, "IQ4_NL": 4.5, "Q3_K_L": 4.3, "Q3_K_M": 3.9,
              "Q3_K_S": 3.5, "Q2_K": 3.4}


def family(model: ModelEntry) -> str:
    """Model id with the quantization token removed: qwen2.5-1.5b-instruct-q4_k_m -> qwen2.5-1.5b-instruct."""
    mid = model.id
    for q in sorted(FILE_TYPE_IDS, key=len, reverse=True):
        mid = re.sub(rf"[-_.]{re.escape(q.lower())}$", "", mid)
    return mid


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def _tps(res: dict, which: str) -> float | None:
    t = res.get("timings") or {}
    v = t.get(f"{which}_per_second")
    if v:
        return round(float(v), 2)
    if which == "predicted":   # fallback when a build does not report timings
        n = (res.get("usage") or {}).get("completion_tokens")
        if n and res.get("ttft_s") is not None and res["total_s"] > res["ttft_s"]:
            return round(n / (res["total_s"] - res["ttft_s"]), 2)
    return None


def measure_model(cfg: Config, model: ModelEntry, *, runs: int, max_tokens: int, warmup: bool = True,
                  log=print) -> dict:
    """Start the runtime for one model, run the prompt set, collect metrics, stop."""
    mcfg = dataclasses.replace(cfg, model=str(model.path), port=free_loopback_port(), host="127.0.0.1")
    rt = Runtime(mcfg, quiet=True)
    result: dict = {"model": model.to_dict(), "family": family(model), "quant": model.quant, "ok": False}
    sampler = None
    try:
        rt.start()
        b = rt.backend
        sampler = PeakSampler(b.pid).start()
        result.update({
            "startup_s": round(rt.state.startup_s, 3),
            "rss_after_load_bytes": current_rss(b.pid),
            "backend_version": b.version, "stub": b.is_stub or model.stub,
            "command": b.command_line(), "log": str(b.log_path),
        })
        url = rt.base_url()
        if warmup:
            chat(url, baseline_messages(), max_tokens=8, extra=NO_CACHE)
        per_run = {"short": [], "rag": []}
        for i in range(runs):
            for name, msgs in (("short", baseline_messages()), ("rag", rag_messages())):
                r = chat(url, msgs, max_tokens=max_tokens, extra=NO_CACHE)
                per_run[name].append({
                    "run": i + 1, "ttft_s": round(r["ttft_s"], 3) if r["ttft_s"] is not None else None,
                    "total_s": round(r["total_s"], 3), "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
                    "completion_tokens": (r["usage"] or {}).get("completion_tokens"),
                    "prompt_n": (r.get("timings") or {}).get("prompt_n"), "prompt_tps": _tps(r, "prompt"), "gen_tps": _tps(r, "predicted"), "fingerprint": r.get("fingerprint"),
                    "text_preview": r["text"][:160],
                })
            log(f"    run {i + 1}/{runs}: short {per_run['short'][-1]['gen_tps']} tok/s, "
                f"rag TTFT {per_run['rag'][-1]['ttft_s']} s")
        sampler.stop()
        peak, source = sampler.peak()
        result["audit"] = socket_audit(b.pid)
        result.update({
            "ok": True, "peak_rss_bytes": peak, "peak_rss_source": source, "runs": per_run,
            "summary": {
                "short_gen_tps": _median(r["gen_tps"] for r in per_run["short"]),
                "short_total_s": _median(r["total_s"] for r in per_run["short"]),
                "rag_prompt_tokens": _median(r["prompt_tokens"] for r in per_run["rag"]),
                "rag_prompt_tps": _median(r["prompt_tps"] for r in per_run["rag"]),
                "rag_ttft_s": _median(r["ttft_s"] for r in per_run["rag"]),
                "rag_gen_tps": _median(r["gen_tps"] for r in per_run["rag"]),
                "rag_total_s": _median(r["total_s"] for r in per_run["rag"]),
            },
        })
    except RuntimeFailure as e:
        result["error"] = e.to_dict()
    except Exception as e:  # keep the matrix going if one model misbehaves
        result["error"] = {"code": "benchmark_failed", "message": f"{type(e).__name__}: {e}", "hint": ""}
    finally:
        if sampler:
            sampler.stop()
        rt.stop()
    return result


# ------------------------------------------------------------ baseline --

def run_baseline(cfg: Config, *, require_offline: bool, out_dir: Path | None = None, log=print) -> tuple[dict, Path]:
    net_before = outbound_network()
    if require_offline and net_before["outbound_reachable"]:
        raise SecurityError("network_not_disabled", "This machine can reach the internet",
                            "Turn on airplane mode / unplug the network (or run scripts/windows/block-network.ps1), "
                            "then run the baseline again. Drop --require-offline to run anyway.",
                            details={"probe": net_before["attempts"][-1]["target"]})
    rt = Runtime(dataclasses.replace(cfg, port=free_loopback_port(), host="127.0.0.1"), quiet=True)
    log("Starting runtime...")
    rt.start()
    try:
        b = rt.backend
        sampler = PeakSampler(b.pid).start()
        rss_loaded = current_rss(b.pid)
        log(f"Ready in {rt.state.startup_s:.2f} s. Sending the hardcoded prompt...")
        log("")
        res = chat(rt.base_url(), baseline_messages(), max_tokens=160,
                   on_token=lambda t: print(t, end="", flush=True))
        print("\n", flush=True)
        sampler.stop()
        peak, source = sampler.peak()
        audit = socket_audit(b.pid)
        record = {
            "kind": "baseline", "runtime_version": __version__, "prompt_set": PROMPT_SET_VERSION,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stub": b.is_stub or rt.state.model.stub,
            "system": system_info(),
            "network": {"before": net_before, "after": outbound_network(), "require_offline": require_offline},
            "socket_audit": audit,
            "backend": {"version": b.version, "command": b.command_line(), "log": str(b.log_path)},
            "config": {k: v for k, v in cfg.to_dict().items() if k not in ("base_dir",)},
            "model": rt.state.model.to_dict(),
            "prompt": baseline_messages()[0]["content"],
            "response": res["text"],
            "metrics": {
                "startup_s": round(rt.state.startup_s, 3),
                "rss_after_load_bytes": rss_loaded,
                "peak_rss_bytes": peak, "peak_rss_source": source,
                "ttft_s": round(res["ttft_s"], 3) if res["ttft_s"] else None,
                "total_s": round(res["total_s"], 3),
                "prompt_tps": _tps(res, "prompt"), "gen_tps": _tps(res, "predicted"),
                "completion_tokens": (res["usage"] or {}).get("completion_tokens"),
            },
        }
    finally:
        rt.stop()
    out_dir = out_dir or cfg.runs_path / f"baseline-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out_dir / "baseline.md").write_text(baseline_markdown(record), encoding="utf-8")
    return record, out_dir


def baseline_markdown(r: dict) -> str:
    m, s, net = r["metrics"], r["system"], r["network"]
    isolated = not net["before"]["outbound_reachable"] and not net["after"]["outbound_reachable"]
    audit = r["socket_audit"]
    lines = [
        "# Week 1 - Runtime baseline", "",
        SIM_BANNER if r["stub"] else "", "",
        f"- **When:** {r['timestamp']}",
        f"- **Machine:** {s['cpu']}, {s['physical_cores']}C/{s['logical_cores']}T, {fmt_bytes(s['ram_total_bytes'])} RAM, {s['os']}",
        f"- **Model:** `{r['model']['id']}` ({r['model']['quant']}, {r['model'].get('display_size') or fmt_bytes(r['model']['size_bytes'])})",
        f"- **llama-server:** {r['backend']['version']}",
        f"- **Network:** {'isolated - no outbound connection possible' if isolated else 'REACHABLE - internet was available during this run'}",
        "- **Backend sockets:** " + (("loopback only (" + ", ".join(audit["listening"]) + ")") if audit.get("available") and audit["loopback_only"]
                                      else ("VIOLATIONS: " + "; ".join(audit.get("violations", [])) if audit.get("available") else audit.get("reason", "n/a"))),
        "", "## Metrics", "",
        "| Metric | Value |", "|---|---|",
        f"| Startup (spawn -> /health 200) | {m['startup_s']:.2f} s |",
        f"| RAM after load | {fmt_bytes(m['rss_after_load_bytes'])} |",
        f"| Peak RAM ({m['peak_rss_source']}) | {fmt_bytes(m['peak_rss_bytes'])} |",
        f"| Time to first token | {m['ttft_s']} s |",
        f"| Prompt processing | {m['prompt_tps']} tok/s |",
        f"| Generation | {m['gen_tps']} tok/s |",
        f"| Tokens generated | {m['completion_tokens']} |",
        "", "## Hardcoded prompt", "", f"> {r['prompt']}", "", "## Response", "", "```", r["response"].strip(), "```",
        "", "## Exact command", "", "```", r["backend"]["command"], "```", "",
        "Full record (config, system, network probes): `baseline.json`. Backend log: `" + r["backend"]["log"] + "`.", "",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------- matrix --

def run_matrix(cfg: Config, models: list[ModelEntry], *, runs: int, max_tokens: int, ram_budget_gb: float,
               min_gen_tps: float = MIN_GEN_TPS, max_ttft_s: float = MAX_RAG_TTFT_S,
               out_dir: Path | None = None, log=print) -> tuple[dict, Path]:
    if not models:
        raise ConfigError("no_models", "No models selected for the benchmark", "Pass --models or put GGUF files in the models folder.")
    results = []
    for i, m in enumerate(models, 1):
        log(f"[{i}/{len(models)}] {m.id} ({m.quant}, {m.display_size})")
        r = measure_model(cfg, m, runs=runs, max_tokens=max_tokens, log=log)
        if r["ok"]:
            s = r["summary"]
            log(f"    startup {r['startup_s']:.2f} s, peak RAM {fmt_bytes(r['peak_rss_bytes'])}, "
                f"gen {s['short_gen_tps']} tok/s, RAG TTFT {s['rag_ttft_s']} s")
        else:
            log(f"    FAILED: {r['error']['message']}")
        results.append(r)
    doc = {
        "kind": "quant_matrix", "runtime_version": __version__, "prompt_set": PROMPT_SET_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "system": system_info(),
        "settings": {"runs": runs, "max_tokens": max_tokens, "ctx_size": cfg.ctx_size, "threads": cfg.threads or "auto",
                     "n_gpu_layers": cfg.n_gpu_layers, "temperature": 0, "ram_budget_gb": ram_budget_gb,
                     "min_gen_tps": min_gen_tps, "max_ttft_s": max_ttft_s},
        "network": outbound_network(),
        "stub": any(r.get("stub") for r in results),
        "results": results,
    }
    doc["analysis"] = analyse(results, ram_budget_gb, min_gen_tps, max_ttft_s)
    out_dir = out_dir or cfg.runs_path / f"bench-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    write_csv(doc, out_dir / "results.csv")
    (out_dir / "report.md").write_text(matrix_markdown(doc), encoding="utf-8")
    return doc, out_dir


def analyse(results: list[dict], ram_budget_gb: float, min_gen_tps: float = MIN_GEN_TPS,
            max_ttft_s: float = MAX_RAG_TTFT_S) -> dict:
    ok = [r for r in results if r["ok"]]
    comparisons = []
    fams: dict[str, list[dict]] = {}
    for r in ok:
        fams.setdefault(r["family"], []).append(r)
    for fam, rs in fams.items():
        if len(rs) < 2:
            continue
        ref = max(rs, key=lambda r: _PRECISION.get(r["quant"] or "", 0))
        for r in rs:
            if r is ref:
                continue
            pct = lambda a, b: round((a - b) / b * 100, 1) if a and b else None  # noqa: E731
            comparisons.append({
                "family": fam, "quant": r["quant"], "vs": ref["quant"],
                "peak_ram_pct": pct(r["peak_rss_bytes"], ref["peak_rss_bytes"]),
                "gen_tps_pct": pct(r["summary"]["short_gen_tps"], ref["summary"]["short_gen_tps"]),
                "rag_ttft_pct": pct(r["summary"]["rag_ttft_s"], ref["summary"]["rag_ttft_s"]),
                "startup_pct": pct(r["startup_s"], ref["startup_s"]),
                "file_size_pct": pct(r["model"].get("weight_bytes") or r["model"]["size_bytes"],
                                     ref["model"].get("weight_bytes") or ref["model"]["size_bytes"]),
            })
    budget = ram_budget_gb * 1024**3
    usable = [r for r in ok if r["peak_rss_bytes"] <= budget
              and (r["summary"]["short_gen_tps"] or 0) >= min_gen_tps
              and (r["summary"]["rag_ttft_s"] or 1e9) <= max_ttft_s]
    recommended = None
    if usable:
        # Quality proxy = bits per weight, capped at Q6_K (above that, gains are negligible for chat use).
        # Among equals prefer the smaller memory footprint, then the faster one.
        best = max(usable, key=lambda r: (min(_PRECISION.get(r["quant"] or "", 5), 6.6), -r["peak_rss_bytes"],
                                          r["summary"]["short_gen_tps"] or 0))
        recommended = {"id": best["model"]["id"], "quant": best["quant"],
                       "reason": f"highest-precision option (bits per weight, capped at Q6_K; ties go to lower RAM) that fits "
                                 f"the {ram_budget_gb:g} GB budget (peak {fmt_bytes(best['peak_rss_bytes'])}) and is usable "
                                 f"(>= {min_gen_tps:g} tok/s generation, <= {max_ttft_s:g} s to first token on the RAG prompt)"}
    return {"comparisons": comparisons, "recommended": recommended,
            "failed": [{"id": r["model"]["id"], "error": r["error"]} for r in results if not r["ok"]]}


CSV_FIELDS = ["model_id", "family", "quant", "file_size_mib", "startup_s", "rss_after_load_mib", "peak_rss_mib",
              "short_gen_tps", "short_total_s", "rag_prompt_tokens", "rag_prompt_tps", "rag_ttft_s", "rag_gen_tps",
              "rag_total_s", "stub", "ok", "error"]


def write_csv(doc: dict, path: Path) -> None:
    mib = lambda b: round(b / 1024**2, 1) if b else None  # noqa: E731
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in doc["results"]:
            s = r.get("summary", {})
            w.writerow({
                "model_id": r["model"]["id"], "family": r["family"], "quant": r["quant"],
                "file_size_mib": mib(r["model"]["size_bytes"]), "startup_s": r.get("startup_s"),
                "rss_after_load_mib": mib(r.get("rss_after_load_bytes")), "peak_rss_mib": mib(r.get("peak_rss_bytes")),
                **{k: s.get(k) for k in CSV_FIELDS if k in s}, "stub": r.get("stub"), "ok": r["ok"],
                "error": (r.get("error") or {}).get("message", ""),
            })


def matrix_markdown(doc: dict) -> str:
    s = doc["system"]
    st = doc["settings"]
    lines = ["# Week 2 - Quantization performance matrix", ""]
    if doc["stub"]:
        lines += [SIM_BANNER, ""]
    lines += [
        f"- **When:** {doc['timestamp']}  |  **Prompt set:** {doc['prompt_set']}  |  **Runtime:** {doc['runtime_version']}",
        f"- **Machine:** {s['cpu']}, {s['physical_cores']}C/{s['logical_cores']}T, {fmt_bytes(s['ram_total_bytes'])} RAM, {s['os']}",
        f"- **Settings:** {st['runs']} runs per prompt (median reported), max_tokens {st['max_tokens']}, ctx {st['ctx_size']}, "
        f"threads {st['threads']}, GPU layers {st['n_gpu_layers']}, temperature 0",
        f"- **Network during run:** {'reachable' if doc['network']['outbound_reachable'] else 'isolated'}",
        "", "## Results", "",
        "| Model | Quant | File | Startup | Peak RAM | Gen tok/s (short) | RAG prompt tok/s | RAG time-to-first-token | RAG total |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in doc["results"]:
        if not r["ok"]:
            lines.append(f"| {r['family']} | {r['quant']} | {r['model'].get('display_size') or fmt_bytes(r['model']['size_bytes'])} | FAILED: {r['error']['code']} | | | | | |")
            continue
        x = r["summary"]
        lines.append(
            f"| {r['family']} | {r['quant']} | {r['model'].get('display_size') or fmt_bytes(r['model']['size_bytes'])} | {r['startup_s']:.2f} s | "
            f"{fmt_bytes(r['peak_rss_bytes'])} | {x['short_gen_tps']} | {x['rag_prompt_tps']} | {x['rag_ttft_s']} s | {x['rag_total_s']} s |"
        )
    a = doc["analysis"]
    if a["comparisons"]:
        lines += ["", "## Compression vs. reference quant (same model)", "",
                  "| Model | Quant vs ref | File size | Peak RAM | Gen speed | RAG time-to-first-token | Startup |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        f = lambda v: "n/a" if v is None else f"{v:+.1f}%"  # noqa: E731
        for c in a["comparisons"]:
            lines.append(f"| {c['family']} | {c['quant']} vs {c['vs']} | {f(c['file_size_pct'])} | {f(c['peak_ram_pct'])} | "
                         f"{f(c['gen_tps_pct'])} | {f(c['rag_ttft_pct'])} | {f(c['startup_pct'])} |")
    lines += ["", "## Recommendation", ""]
    if a["recommended"]:
        rec = a["recommended"]
        lines.append(f"Default for a {st['ram_budget_gb']:g} GB RAM budget: **`{rec['id']}`** - {rec['reason']}.")
    else:
        lines.append(f"No model fits the {st['ram_budget_gb']:g} GB RAM budget while staying usable "
                     f"(>= {st.get('min_gen_tps', MIN_GEN_TPS):g} tok/s, <= {st.get('max_ttft_s', MAX_RAG_TTFT_S):g} s "
                     "to first token on the RAG prompt). "
                     "Try a smaller model or a lower-bit quantization.")
    lines += ["",
              "Answer quality is not measured here. Lower-bit quantizations are faster and smaller but can lose accuracy; "
              "the final pick should be confirmed against E5's grounding / citation eval set.", "",
              "## How to read this", "",
              "- **Startup**: process spawn until `/health` returns 200 (model loaded, ready to answer).",
              "- **Peak RAM**: OS peak working set of llama-server (Windows `PeakWorkingSetSize`, Linux `VmHWM`). "
              "With mmap enabled, model pages count once they are touched.",
              "- **Gen tok/s**: tokens generated per second as reported by llama.cpp `timings.predicted_per_second`.",
              "- **RAG prompt tok/s / time-to-first-token**: how fast the model reads ~1.3k tokens of retrieved passages - "
              "on CPU this is usually the wait the user notices most in document Q&A.", "",
              "Raw per-run data: `results.json`; spreadsheet-friendly: `results.csv`.", ""]
    if a["failed"]:
        lines += ["## Failures", ""] + [f"- `{x['id']}`: [{x['error']['code']}] {x['error']['message']} - {x['error'].get('hint', '')}" for x in a["failed"]] + [""]
    return "\n".join(lines)


def models_for_bench(cfg: Config, selectors: list[str] | None) -> list[ModelEntry]:
    models, _ = list_models(cfg.models_path)
    if not selectors:
        return models
    chosen = []
    for sel in selectors:
        p = cfg.resolve(sel)
        if p.is_file():
            chosen.append(ModelEntry.from_path(p))
            continue
        hits = [m for m in models if sel.lower() in m.id]
        if not hits:
            raise ConfigError("model_not_found", f"No model matches '{sel}'",
                              "Available: " + (", ".join(m.id for m in models) or "(none)"))
        chosen.extend(h for h in hits if h not in chosen)
    return chosen
