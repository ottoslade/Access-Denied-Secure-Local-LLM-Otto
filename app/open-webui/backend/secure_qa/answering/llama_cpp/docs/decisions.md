# Design decisions: E1 runtime

## 1. Wrap the official `llama-server` rather than embed llama.cpp

We launch the upstream binary as a child process and talk to it over loopback HTTP.

* Upstream ships Windows builds (CPU, Vulkan, CUDA) weekly. Updating means replacing `bin\`, with no rebuild of our code.
* A crash in native code (bad model file, out of memory) kills the child, not the launcher. The launcher reports it
  with a diagnosis and the relevant log lines.
* The alternative (`llama-cpp-python` in-process) is easier to bundle into one EXE, but lags upstream by weeks and
  ties us to a Python/C++ ABI.

## 2. A thin gateway in front of llama-server

The public port (`8080`) is ours. llama-server sits on a random internal port. The gateway gives us:

* **A stable contract.** llama.cpp changes its response fields and flags often. E3/E4 code against `docs/api.md`,
  not against whichever llama.cpp version is installed.
* **Health with a reason** (`loading`, `error` plus hint) instead of a bare 503.
* **Actionable errors** with a stable `code` and a user-facing `hint`.
* **Security controls**; see §4.
* **Privacy.** The access log records method, path, status and latency only. Prompt and document text is never
  logged (a test enforces this). llama-server's own `/v1/models` exposes the model's full file path; ours doesn't.

The cost is one extra loopback hop: about 1 ms per request, which is negligible next to inference.

## 3. Standard library + `psutil` only

The runtime uses Python's standard library (`http.server`, `http.client`, `tomllib`) plus `psutil` for memory and
socket measurement. There's no web framework and nothing to audit or update on an air-gapped machine. PyInstaller
bundle size is about 19 MB.

## 4. Offline and local-only guarantees

| Control | Where |
|---|---|
| Gateway binds `127.0.0.1` and refuses anything else unless `allow_remote = true` | `offline.require_loopback` |
| llama-server always gets `--host 127.0.0.1`, a random free port, and `--offline` when the build supports it | `backend.build_args` |
| Environment scrubbed before spawn: `LLAMA_ARG_*`, `HF_*`, proxies removed (so nothing can trigger an `-hf` download) | `discovery.scrubbed_env` |
| **Per-launch random API key on llama-server**, passed via environment (never on the command line or in logs). llama-server enables CORS for all origins by default, so without this any web page could call the internal port directly | `backend.start` |
| Gateway rejects non-local `Host` (DNS rebinding) and non-local `Origin` | `gateway._guard` |
| 16 MiB request cap | `gateway.MAX_BODY` |
| Evidence: `baseline` records outbound-network probes and a socket audit of llama-server | `offline.outbound_network`, `socket_audit` |
| Optional Windows Firewall block rules for both executables | `scripts/windows/block-network.ps1` |

## 5. Process lifetime

On Windows, llama-server is placed in a **Job Object with KILL_ON_JOB_CLOSE**, so it dies even if the launcher is
killed from Task Manager. Otherwise a 3 GB process would be left holding the port. Elsewhere an `atexit` hook
stops it. Linux's `PR_SET_PDEATHSIG` was tried and rejected: it fires when the spawning *thread* exits, which killed
the backend when the runtime was started from a worker thread. A test covers this case.

## 6. Measurement methodology

* **Startup:** process spawn until `/health` returns 200, polled every 50 ms.
* **Peak RAM:** the OS lifetime peak counter (Windows `PeakWorkingSetSize`, Linux `VmHWM`), cross-checked with
  100 ms RSS sampling. The report says which source was used.
* **Tokens/s:** llama.cpp's own `timings` (engine-side), so HTTP overhead isn't counted.
* **Two prompts:** a short one (generation speed), and a RAG-shaped one with about 1.3k tokens of synthetic
  passages, because on CPU reading the retrieved context is usually the longest wait.
* **Prompt cache disabled** (`cache_prompt: false`). Otherwise llama.cpp reuses the identical prompt from the
  previous run and prompt speed looks near-instant. (The first real-engine test run caught exactly this.)
* Temperature 0, fixed seed, warm-up request excluded, median of N runs.
* **Recommendation rule:** among options that fit the RAM budget and meet the speed floors (≥ 8 tok/s,
  ≤ 15 s to first token by default), pick the highest bits-per-weight, capped at Q6_K, with ties going to lower RAM.
  Answer quality is **not** measured. E5's eval set must confirm the pick.

## 7. Stub backend

`--server-bin stub` runs a stand-in llama-server that speaks the same CLI and HTTP API. It **allocates real memory**
in proportion to the simulated model/quant, so the RAM measurement path is exercised for real. Speed is modelled as
memory-bandwidth bound. Every stub output is labelled (`system_fingerprint: docqa-stub`, `engine.stub: true`,
report banners), so simulated numbers can't be mistaken for real ones. It also has fault injection
(`DOCQA_STUB_FAIL=load|hang_load|crash_after_first`) for tests.

## Verification done in this workspace

* 56 automated tests (unit, HTTP integration, failure modes, CLI exit codes, security): 55 run against the stub, plus one opt-in test against a real llama-server.
* The same launcher, gateway and benchmark run against a **real `llama-server` built from llama.cpp source**, using
  a tiny random-weight model built from llama.cpp's bundled vocab and quantized to F32/F16/Q8_0/Q4_K_M.
  This confirmed flag compatibility, `--offline`, `/props` context capping, the `timings` format, the
  context-overflow error, the backend API key, and the corrupt-model and bad-flag diagnoses.
* The PyInstaller spec builds and the frozen bundle runs, finding its config and folders relative to the executable.

## Known limitations / not verified here

* **Not run on Windows.** The Windows-only pieces (Job Object, `PeakWorkingSetSize`, the PowerShell scripts,
  `.exe` build) follow documented APIs but need a first run on the reference laptop.
* No real-model numbers yet. `docs/sample-results` contains stub data and tiny-random-model data only.
* One model per runtime instance. Switching models means a restart (`--model`).
* The model shortlist's Hugging Face repo and file names should be verified before downloading.
