# Week 2 Friday demo runbook

**PM acceptance:** "Run one command, show endpoints, send a chat request, and compare at least two compression levels."

## Before the meeting (on a connected machine)

```powershell
scripts\windows\fetch_runtime.ps1                # llama.cpp -> bin\
scripts\windows\fetch_models.ps1 -Set week2      # Qwen3-4B Q4_K_M + Q8_0 -> models\  (~7 GB)
scripts\windows\build_exe.ps1 -IncludeBin -IncludeModels -Zip
```

Copy `dist\DocQA-Runtime\` to the reference laptop. Run `docqa-runtime bench` once in advance: it takes about
5–15 minutes on CPU, and you don't want to wait on it live. Keep the `report.md`.

## Live (about 5 minutes)

Turn on airplane mode, then run `scripts\demo.ps1` from the bundle, or do the steps by hand:

| # | Do | Show | Definition of Done item |
|---|---|---|---|
| 1 | `docqa-runtime doctor` | All OK, "no outbound network (offline)" | environment is ready |
| 2 | `docqa-runtime up` | Banner with the three endpoints and "ready in N s" | *fresh launch prints endpoints* |
| 3 | Browser: `http://127.0.0.1:8080/health` | `"status": "ok"`, model and quant, `loopback_only: true` | *health passes* |
| 4 | `docqa-runtime chat "Why does running locally keep documents private?"` | Streaming answer and tok/s | *sample request succeeds* |
| 5 | `docqa-runtime chat hi --url http://127.0.0.1:9999` | `runtime_not_running` with a hint | *errors are actionable* |
| 6 | Rename a model, then `docqa-runtime up --model missing` | `model_not_found` listing the available ids | *errors are actionable* |
| 7 | Open `runs\bench-…\report.md` | Q4_K_M vs Q8_0: RAM, startup, tok/s, time-to-first-token and a recommendation | *compare at least two compression levels* |

**Without the real models** (e.g. rehearsing on another PC): `scripts\demo.ps1 -Stub` runs the whole flow on the
simulated backend. Every screen is labelled STUB / SIMULATED.

## If something goes wrong

| Message | Fix |
|---|---|
| `server_not_found` | `bin\llama-server.exe` missing. Run `fetch_runtime.ps1` or copy the bin folder |
| `port_in_use` | Another copy is running. Check `http://127.0.0.1:8080/health`, or use `--port 8090` |
| `insufficient_ram` / `backend_out_of_memory` | Use the Q4_K_M file, or `--ctx-size 2048` |
| `model_load_failed` | File is incomplete. Compare its size and SHA256 with `SHA256SUMS.txt` |
| `backend_bad_argument` | llama.cpp build too old/new for a flag in `extra_args`. Remove the flag |
| Windows SmartScreen blocks the exe | "More info" → "Run anyway", or right-click → Properties → Unblock |
