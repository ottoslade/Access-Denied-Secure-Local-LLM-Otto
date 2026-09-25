# docqa-runtime: E1 LLM runtime for Secure Local DocQA

One command starts a local LLM (llama.cpp) behind a small, stable, **loopback-only** API that the rest of the
app talks to. It also produces the Week 1 baseline record and the Week 2 quantization comparison.

```
docqa-runtime up

DocQA runtime ready in 3.84 s
  Model     qwen_qwen3-4b-instruct-2507-q4_k_m  (Q4_K_M, 2.33 GiB, ctx 4096)
  Health    GET  http://127.0.0.1:8080/health
  Models    GET  http://127.0.0.1:8080/v1/models
  Chat      POST http://127.0.0.1:8080/v1/chat/completions
  Engine    llama-server 6543 (a1b2c3d)  (pid 4312, internal port 127.0.0.1:53211)
  Offline   on; bound to 127.0.0.1 (this computer only)
```

| Plan item | Where |
|---|---|
| **W1** Shortlist compact text-only GGUF models | [`docs/week1-candidate-models.md`](docs/week1-candidate-models.md), `models/shortlist.json` |
| **W1** llama.cpp launch, hardcoded prompt with networking disabled, command/config recorded, peak RAM + tok/s | `docqa-runtime baseline --require-offline` → `runs/baseline-*/baseline.md` |
| **W2** One-command launcher/CLI + EXE | `docqa-runtime up`, `scripts/windows/build_exe.ps1` |
| **W2** `/health`, `/v1/models`, `/v1/chat/completions` | [`docs/api.md`](docs/api.md) (contract for E3/E4) |
| **W2** Actionable errors | every failure has `code` + `hint` + exit code (see api.md) |
| **W2** Quantization matrix (RAM / startup / tokens-per-second) | `docqa-runtime bench` → `runs/bench-*/report.md` |
| Friday demo | [`docs/week2-demo-runbook.md`](docs/week2-demo-runbook.md), `scripts/windows/demo.ps1` |
| Design and security choices | [`docs/decisions.md`](docs/decisions.md) |

## Try it in 1 minute (no model needed)

On Windows, double-click **`start.cmd`** (or run it from a terminal). It sets up `.venv` on first run, starts
the runtime and lets you chat with it in the same window (`exit` stops it). It uses the stub backend automatically
until llama.cpp and a model are in `bin\` and `models\`. `start.cmd --server-only` starts just the API.
Or by hand:

The **stub backend** stands in for llama.cpp and speaks the same API. Everything it produces is labelled STUB / SIMULATED.

```bash
pip install -e .                 # Python 3.11+
docqa-runtime stub-models        # writes placeholder GGUF headers to models/
docqa-runtime up --server-bin stub
# in another terminal
docqa-runtime chat "Hello, are you running locally?"
curl http://127.0.0.1:8080/health
```

## Real setup (Windows reference laptop)

On a machine with internet:

```powershell
scripts\windows\fetch_runtime.ps1              # official llama.cpp Windows CPU build -> bin\
scripts\windows\fetch_models.ps1 -Set week2    # Qwen3-4B-Instruct-2507 at Q4_K_M and Q8_0 -> models\
scripts\windows\build_exe.ps1 -IncludeBin -IncludeModels -Zip   # -> dist\DocQA-Runtime.zip
```

Copy `DocQA-Runtime\` to the laptop (no Python needed there), disconnect the network, then:

```powershell
docqa-runtime doctor                       # checks binary, model, RAM, port, network
docqa-runtime baseline --require-offline   # Week 1 record
docqa-runtime up                           # or double-click start-runtime.cmd
docqa-runtime bench                        # Week 2 matrix: every .gguf in models\
```

## Commands

| Command | What it does |
|---|---|
| `up` (default) | Start gateway + llama-server. `--chat` to talk to it in the same window, `--json` / `--ready-file` for the desktop app |
| `doctor` | Pre-flight check without starting anything |
| `models` | List GGUF files with quant, size and architecture. `*` marks the default pick |
| `chat "…"` | Ask the running runtime (streams the answer, prints tok/s). No question = interactive conversation |
| `baseline` | Week 1: fixed prompt, startup, RAM after load, peak RAM, tok/s, network probes, socket audit |
| `bench` | Week 2: for each model: startup, peak RAM, generation tok/s, RAG-prompt read speed and time-to-first-token (median of `--runs`), comparison vs the highest-precision quant, and a recommendation for `--ram-budget-gb` |
| `stub-models` | Create placeholder models for the stub backend |

Common flags: `--model <id|file|path>`, `--models-dir`, `--server-bin <path|stub>`, `--port`, `--ctx-size`,
`--threads`, `--config`. Every flag has a `runtime.toml` equivalent, so the packaged folder works with no flags.

## Layout

```
src/docqa_runtime/
  cli.py          commands            launcher.py  up / Runtime (gateway + backend lifecycle)
  gateway.py      public API          backend.py   llama-server process, diagnosis of failures
  discovery.py    find binary/models  gguf.py      GGUF header reader (quant, arch, ctx)
  bench.py        baseline + matrix   prompts.py   fixed benchmark prompts (versioned)
  offline.py      loopback/network    metrics.py   peak RAM, system info
  stub_server.py  simulated llama-server             client.py    stdlib API client
scripts/windows/  fetch_runtime, fetch_models, build_exe, block-network, demo (PowerShell)
scripts/dev/      make_tiny_gguf.py (random-weight model for testing the real engine)
packaging/        PyInstaller spec, start-runtime.cmd, README-RUN.txt
docs/             api, week1 models, week2 runbook, decisions, sample-results/
tests/            56 tests
```

## Tests

```bash
pip install -e ".[dev]" && pytest          # about 1 minute, uses the stub
# against a real llama-server + a folder of GGUFs:
DOCQA_TEST_LLAMA_SERVER=/path/llama-server DOCQA_TEST_MODELS_DIR=/path/models pytest -k real
```

## Status

Built and tested on Linux against the stub **and** a real `llama-server` compiled from llama.cpp source (with a
tiny random-weight model). **Not yet run on Windows or with real model weights.** The first run on the reference
laptop is the next step; see "Known limitations" in `docs/decisions.md`. The numbers in `docs/sample-results/` show
the report format only.
