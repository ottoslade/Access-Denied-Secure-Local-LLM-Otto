# Week 1 - Runtime baseline

> **SIMULATED RESULTS - stub backend.** These numbers come from the stub llama-server, not a real model. They prove the pipeline works end to end; re-run on the reference laptop with real GGUF files for real numbers.

- **When:** 2026-09-22T23:35:31-0400
- **Machine:** Intel(R) Xeon(R) Processor @ 2.10GHz, 2C/2T, 7.84 GiB RAM, Linux 6.18.44-fc-v37 (#1 SMP PREEMPT_DYNAMIC @0)
- **Model:** `stub-1b-instruct-q4_k_m` (Q4_K_M, 583 MiB simulated)
- **llama-server:** 0 (docqa-stub)
- **Network:** REACHABLE - internet was available during this run
- **Backend sockets:** loopback only (127.0.0.1:42975)

## Metrics

| Metric | Value |
|---|---|
| Startup (spawn -> /health 200) | 0.84 s |
| RAM after load | 738 MiB |
| Peak RAM (os_peak_counter) | 738 MiB |
| Time to first token | 0.184 s |
| Prompt processing | 217.67 tok/s |
| Generation | 24.4 tok/s |
| Tokens generated | 160 |

## Hardcoded prompt

> In two sentences, explain what a private, offline document question-answering assistant does.

## Response

```
This is a simulated answer from the DocQA stub runtime. No model weights were loaded, so the text carries no meaning, but its length, pacing and response format match what the real llama.cpp server would return for the same request. This is a simulated answer from the DocQA stub runtime. No model weights were loaded, so the text carries no meaning, but its length, pacing and response format match what the real llama.cpp server would return for the same request. This is a simulated answer from the DocQA stub runtime. No model weights were loaded, so the text carries no meaning, but its length, pacing and response format match what the real llama.cpp server would return for the same request. This is a simulated answer from the DocQA stub runtime. No model weights were loaded, so the text carries no meaning, but its length, pacing and response format match what the real llama.cpp server would return for the same request.
```

## Exact command

```
python -m docqa_runtime.stub_server -m ./models/stub-1b-instruct-Q4_K_M.gguf --alias stub-1b-instruct-q4_k_m --host 127.0.0.1 --port 42975 -c 4096 -t 2 -ngl 0 --offline --no-webui
```

Full record (config, system, network probes): `baseline.json`. Backend log: `./runs/logs/llama-server-20260922-233523-stub-1b-instruct-q4_k_m.log`.
