"""Command line entry point.

    docqa-runtime [up]          start the runtime (default command)
    docqa-runtime models        list GGUF models found on disk
    docqa-runtime doctor        check the setup without starting anything
    docqa-runtime chat "..."    ask the running runtime a question
    docqa-runtime baseline      Week 1: hardcoded prompt + startup/RAM/tokens-per-second record
    docqa-runtime bench         Week 2: quantization performance matrix
    docqa-runtime stub-models   create stub model files for a dry run without real weights
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .errors import EXIT_CONFIG, EXIT_OK, RuntimeFailure

COMMANDS = {"up", "models", "doctor", "chat", "baseline", "bench", "stub-models", "_stub-server", "-h", "--help", "--version"}


def _runtime_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("runtime options (override runtime.toml)")
    g.add_argument("--config", help="path to runtime.toml (default: ./runtime.toml or next to the program)")
    g.add_argument("--model", help="model file path, file name, or id (see `docqa-runtime models`)")
    g.add_argument("--models-dir", help="folder with .gguf files (default: models)")
    g.add_argument("--server-bin", help="path to llama-server(.exe), or 'stub' for the simulated backend")
    g.add_argument("--host", help="gateway bind address (default 127.0.0.1)")
    g.add_argument("--port", type=int, help="gateway port (default 8080)")
    g.add_argument("--ctx-size", type=int, help="context window in tokens (default 4096)")
    g.add_argument("--threads", type=int, help="CPU threads (0 = physical cores)")
    g.add_argument("--gpu-layers", type=int, dest="n_gpu_layers", help="layers to offload to GPU (default 0)")
    g.add_argument("--allow-remote", action="store_true", default=None, help="permit binding to a non-loopback address")


def _overrides(a) -> dict:
    keys = ("model", "models_dir", "server_bin", "host", "port", "ctx_size", "threads", "n_gpu_layers", "allow_remote")
    return {k: getattr(a, k, None) for k in keys}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docqa-runtime", description="Secure Local DocQA - local LLM runtime (llama.cpp)")
    p.add_argument("--version", action="version", version=f"docqa-runtime {__version__}")
    sub = p.add_subparsers(dest="cmd")

    up = sub.add_parser("up", help="start the runtime (default)")
    _runtime_flags(up)
    up.add_argument("--json", action="store_true", help="print one JSON line when ready instead of the banner")
    up.add_argument("--ready-file", help="write endpoint info as JSON to this file when ready (for the desktop app)")

    m = sub.add_parser("models", help="list models on disk")
    _runtime_flags(m)
    m.add_argument("--json", action="store_true")

    d = sub.add_parser("doctor", help="check the setup without starting the runtime")
    _runtime_flags(d)

    c = sub.add_parser("chat", help="send one question to a running runtime")
    c.add_argument("prompt", nargs="+")
    c.add_argument("--url", default="http://127.0.0.1:8080")
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--system", help="optional system prompt")
    c.add_argument("--no-stream", action="store_true")

    b = sub.add_parser("baseline", help="Week 1 baseline: hardcoded prompt + metrics record")
    _runtime_flags(b)
    b.add_argument("--require-offline", action="store_true", help="fail if the machine can reach the internet")
    b.add_argument("--out", help="output folder (default runs/baseline-<timestamp>)")

    q = sub.add_parser("bench", help="Week 2 quantization matrix")
    _runtime_flags(q)
    q.add_argument("--models", nargs="*", help="model ids / name fragments / paths (default: every model in the folder)")
    q.add_argument("--runs", type=int, default=3, help="runs per prompt; the median is reported (default 3)")
    q.add_argument("--max-tokens", type=int, default=128)
    q.add_argument("--ram-budget-gb", type=float, default=6.0, help="RAM budget used for the recommendation (default 6)")
    q.add_argument("--min-gen-tps", type=float, default=8.0, help="speed floor for the recommendation (default 8 tok/s)")
    q.add_argument("--max-ttft-s", type=float, default=15.0,
                   help="max time-to-first-token on the RAG prompt for the recommendation (default 15 s)")
    q.add_argument("--out", help="output folder (default runs/bench-<timestamp>)")

    s = sub.add_parser("stub-models", help="create stub model files for a dry run")
    s.add_argument("--out", default=None, help="folder (default: the configured models folder)")
    s.add_argument("--config")
    s.add_argument("--params-b", type=float, default=1.0, help="simulated parameter count in billions (default 1.0)")
    s.add_argument("--quants", nargs="*", default=None, help="default: Q4_K_M Q5_K_M Q8_0 F16")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_stub-server":            # used by the frozen .exe to run the stub backend
        from .stub_server import main as stub_main
        return stub_main(argv[1:])
    if not argv or argv[0] not in COMMANDS:
        argv = ["up", *argv]                          # `docqa-runtime` alone == `docqa-runtime up`
    args = build_parser().parse_args(argv)
    try:
        return dispatch(args)
    except RuntimeFailure as e:
        print(e.render(), file=sys.stderr, flush=True)
        return e.exit_code
    except KeyboardInterrupt:
        return 130


def dispatch(a) -> int:
    from . import config

    if a.cmd == "chat":
        return cmd_chat(a)
    if a.cmd == "stub-models":
        from .stubmodels import create_stub_models

        cfg = config.load(a.config)
        out = cfg.resolve(a.out) if a.out else cfg.models_path
        paths = create_stub_models(out, n_params=a.params_b * 1e9, quants=a.quants)
        for p in paths:
            print(f"created {p}")
        print("\nStub models contain no weights. Start with:  docqa-runtime up --server-bin stub")
        return EXIT_OK

    cfg = config.load(getattr(a, "config", None), _overrides(a))
    if a.cmd == "up":
        from .launcher import run_up

        return run_up(cfg, json_output=a.json, ready_file=a.ready_file)
    if a.cmd == "models":
        return cmd_models(cfg, a.json)
    if a.cmd == "doctor":
        return cmd_doctor(cfg)
    if a.cmd == "baseline":
        from .bench import run_baseline

        rec, out = run_baseline(cfg, require_offline=a.require_offline, out_dir=cfg.resolve(a.out) if a.out else None)
        m = rec["metrics"]
        from .metrics import fmt_bytes

        print(("[SIMULATED - stub backend]\n" if rec["stub"] else "") +
              f"startup {m['startup_s']:.2f} s | peak RAM {fmt_bytes(m['peak_rss_bytes'])} | "
              f"generation {m['gen_tps']} tok/s | prompt {m['prompt_tps']} tok/s | "
              f"network {'reachable' if rec['network']['before']['outbound_reachable'] else 'isolated'}")
        print(f"Record written to {out}")
        return EXIT_OK
    if a.cmd == "bench":
        from .bench import models_for_bench, run_matrix

        models = models_for_bench(cfg, a.models)
        doc, out = run_matrix(cfg, models, runs=a.runs, max_tokens=a.max_tokens, ram_budget_gb=a.ram_budget_gb,
                              min_gen_tps=a.min_gen_tps, max_ttft_s=a.max_ttft_s, out_dir=cfg.resolve(a.out) if a.out else None)
        print(f"\nReport: {out / 'report.md'}")
        failed = doc["analysis"]["failed"]
        return EXIT_OK if not failed else 1
    return EXIT_CONFIG


def cmd_models(cfg, as_json: bool) -> int:
    from .discovery import list_models, pick_default

    models, problems = list_models(cfg.models_path)
    if as_json:
        print(json.dumps({"models_dir": str(cfg.models_path), "models": [m.to_dict() for m in models],
                          "unreadable": [{"path": str(p), "reason": r} for p, r in problems]}, indent=2))
        return EXIT_OK
    print(f"Models in {cfg.models_path}:")
    if not models:
        print("  (none) - copy .gguf files here, or run `docqa-runtime stub-models` for a dry run")
    default = pick_default(models, cfg.quant_preference).id if models else None
    for m in models:
        flag = "*" if m.id == default else " "
        print(f" {flag} {m.id:<48} {m.quant or '?':<8} {m.display_size:>18}  {m.architecture or '?'}"
              + ("  [stub]" if m.stub else ""))
    for p, r in problems:
        print(f"   ! {p.name}: {r}")
    if default:
        print("\n* = picked when no --model is given")
    return EXIT_OK


def cmd_doctor(cfg) -> int:
    from .discovery import check_resources, find_server, select_model, server_help, server_version
    from .metrics import fmt_bytes, system_info
    from .offline import is_loopback, outbound_network, port_free

    fails = 0

    def line(level, msg):
        nonlocal fails
        fails += level == "FAIL"
        print(f"[{level:<4}] {msg}")

    si = system_info()
    print(f"docqa-runtime {__version__} on {si['os']}, {si['cpu']}, {si['physical_cores']} cores, "
          f"{fmt_bytes(si['ram_total_bytes'])} RAM\n")
    line("OK", f"config: {cfg.source}")
    line("OK" if is_loopback(cfg.host) else ("WARN" if cfg.allow_remote else "FAIL"),
         f"gateway bind address {cfg.host}" + ("" if is_loopback(cfg.host) else " is not loopback"))
    cmd = None
    try:
        cmd = find_server(cfg)
        ver = server_version(cmd)
        help_text = server_help(cmd)
        line("OK", f"llama-server: {cmd[-1] if len(cmd) == 1 else ' '.join(cmd)} (version {ver})")
        if cfg.offline:
            line("OK" if "--offline" in help_text else "WARN",
                 "llama-server supports --offline" if "--offline" in help_text else
                 "this llama-server build has no --offline flag (older build); the launcher still scrubs download settings")
    except RuntimeFailure as e:
        line("FAIL", e.message + (f"\n        -> {e.hint}" if e.hint else ""))
    try:
        m = select_model(cfg)
        line("OK", f"model: {m.id} ({m.quant}, {m.display_size})" + (" [stub]" if m.stub else ""))
        if m.stub and cmd and len(cmd) == 1:
            line("WARN", "a stub model is selected but the real llama-server is configured; use --server-bin stub")
        for w in check_resources(m, cfg):
            line("WARN", w)
    except RuntimeFailure as e:
        line("FAIL", e.message + (f"\n        -> {e.hint}" if e.hint else ""))
    line("OK" if port_free(cfg.host, cfg.port) else "FAIL",
         f"gateway port {cfg.port} " + ("is free" if port_free(cfg.host, cfg.port) else "is in use (runtime already running? use --port)"))
    net = outbound_network()
    line("WARN" if net["outbound_reachable"] else "OK",
         "internet is reachable - fine for setup, but turn networking off for the offline demo"
         if net["outbound_reachable"] else "no outbound network (offline)")
    print(f"\n{'All checks passed.' if not fails else f'{fails} check(s) failed.'}")
    return EXIT_OK if not fails else EXIT_CONFIG


def cmd_chat(a) -> int:
    from .client import ClientError, chat, get_json

    url = a.url.rstrip("/")
    try:
        status, h = get_json(url, "/health", timeout=5)
    except OSError:
        print(f"ERROR [runtime_not_running] Nothing is answering at {url}\n  -> Start it with `docqa-runtime up` "
              "(in another window), or pass --url if it runs on another port.", file=sys.stderr)
        return 3
    if status != 200:
        err = h.get("error") or {}
        print(f"ERROR [runtime_not_ready] runtime status is '{h.get('status')}'"
              + (f": {err.get('message')}\n  -> {err.get('hint')}" if err else "\n  -> Wait until the model has loaded."),
              file=sys.stderr)
        return 3
    msgs = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": " ".join(a.prompt)}]
    try:
        res = chat(url, msgs, max_tokens=a.max_tokens, stream=not a.no_stream,
                   on_token=(lambda t: print(t, end="", flush=True)) if not a.no_stream else None)
    except ClientError as e:
        print(f"\nERROR {e}", file=sys.stderr)
        return 1
    if a.no_stream:
        print(res["text"], end="")
    t = res.get("timings") or {}
    tag = "  [stub]" if res.get("fingerprint") == "docqa-stub" else ""
    print(f"\n\n-- {res.get('model')}{tag} | {(res.get('usage') or {}).get('completion_tokens', '?')} tokens | "
          f"{t.get('predicted_per_second', 0):.1f} tok/s | total {res['total_s']:.2f} s")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
