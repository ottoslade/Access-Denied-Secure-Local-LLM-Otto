# E1 LLM runtime: Weeks 1–2 report (Otto Slade)

**Deliverable:** the `docqa-runtime` project at [`app/open-webui/backend/secure_qa/answering/llama_cpp/`](../../app/open-webui/backend/secure_qa/answering/llama_cpp/README.md). Start with its README; the `docs/` paths below are relative to that folder.

**Status:** Built from scratch and tested on Linux against a stub backend **and a real llama.cpp `llama-server`**.
It has **not yet been run on the Windows reference laptop or with real model weights**; that's the remaining step
for both weeks' Definition of Done.

## Against the plan

| Week | Definition of Done | State |
|---|---|---|
| 1 | Model answers a hardcoded prompt with networking disabled | `docqa-runtime baseline --require-offline` refuses to run if the internet is reachable, then sends the fixed prompt. **Works on the real engine; needs the laptop + a real model.** |
| 1 | Command and config are recorded | Exact llama-server command line, full config, machine specs and llama.cpp version go into `baseline.json` / `.md` |
| 1 | Peak RAM and tokens/sec are captured | OS peak working set, RAM after load, prompt and generation tok/s, time to first token |
| 1 | Candidate matrix | 5 shortlisted models with licence, size, context and role (`docs/week1-candidate-models.md`). Primary: Qwen3-4B-Instruct-2507 (Apache-2.0) |
| 2 | Fresh launch prints endpoints | `docqa-runtime up` banner (or `--json` / `--ready-file` for the desktop app) |
| 2 | Health passes | `GET /health`: 200 when ready, 503 with `loading` / `error` + reason otherwise |
| 2 | Sample request succeeds | `POST /v1/chat/completions`, OpenAI-compatible, streaming and non-streaming. `docqa-runtime chat` for demos |
| 2 | Errors are actionable | Stable `code` + user-facing `hint` + exit code for every failure (missing binary/model, corrupt file, port in use, low RAM, bad flag, context overflow, crash, timeout) |
| 2 | Results compare RAM / startup / tokens-sec | `docqa-runtime bench`: median of N runs, a short prompt plus a ~1.3k-token RAG prompt, comparison vs the highest-precision quant, and a recommendation for a RAM budget |
| 2 | CLI/EXE prototype | PyInstaller bundle (~19 MB) built and smoke-tested (Linux build). Windows build via `scripts/windows/build_exe.ps1` |

## Things worth knowing

- **Security fix beyond the spec.** llama-server enables CORS for every origin by default. Without a fix, any web
  page open in the user's browser could call the model's internal port directly. The launcher gives llama-server a
  random per-launch API key that only the gateway holds, and passes it through the environment so it never appears in
  logs. The gateway also blocks DNS-rebinding and non-local origins, and never logs prompt or document text.
- **Benchmark pitfall avoided.** llama.cpp caches the previous prompt, so repeat runs make reading the documents
  look instant. Benchmarks disable the cache. The first real-engine run exposed this.
- **For E3 (retrieval) and E4 (frontend):** the API contract is in `docs/api.md`. It's worth sharing now so they
  can code against it. A 400 `context_length_exceeded` tells E3 to send fewer or shorter chunks.
- **For E5 (QA/security):** the baseline records outbound-network probes and a socket audit showing llama-server
  listens on loopback only. The recommendation rule doesn't measure answer quality, so E5's eval set should confirm
  the final quantization.

## Sample numbers: format only

**Simulated (stub), 1B model.** These show the report shape. The recommendation follows the rule in
`docs/decisions.md`:

| Quant | Peak RAM | Gen tok/s | RAG time-to-first-token |
|---|---:|---:|---:|
| F16 | 2.01 GiB | 7.5 | 15.8 s |
| Q8_0 | 1.14 GiB | 14.1 | 8.8 s |
| Q5_K_M | 833 MiB | 21.0 | 5.9 s |
| Q4_K_M | 738 MiB | 24.5 | 4.9 s |

**Real llama.cpp, tiny random-weight test model (19M params).** This proves real quantized files are loaded and
measured:

| Quant | File | Peak RAM | Gen tok/s |
|---|---:|---:|---:|
| F32 | 76 MiB | 133 MiB | 612 |
| F16 | 38 MiB | 101 MiB | 818 |
| Q8_0 | 21 MiB | 83 MiB | 1279 |
| Q4_K_M | 13 MiB | 73 MiB | 1214 |

## Next steps on the reference laptop (about 1 hour, mostly downloads)

1. On a connected PC: `fetch_runtime.ps1`, `fetch_models.ps1 -Set week2`, `build_exe.ps1 -IncludeBin -IncludeModels -Zip`.
2. On the laptop, with networking off: `docqa-runtime doctor`, then `baseline --require-offline`, then `bench`.
3. Check the Hugging Face file names in `models/shortlist.json` before step 1. They follow the usual naming but
   haven't been verified from here, because this workspace couldn't reach Hugging Face.
4. Watch for Windows-only issues on the first run: the Job Object cleanup, the PowerShell scripts, and SmartScreen
   on the unsigned exe.
