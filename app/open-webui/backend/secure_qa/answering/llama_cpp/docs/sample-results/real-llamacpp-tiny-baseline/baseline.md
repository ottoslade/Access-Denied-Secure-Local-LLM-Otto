# Week 1 - Runtime baseline



- **When:** 2026-09-22T23:35:31-0400
- **Machine:** Intel(R) Xeon(R) Processor @ 2.10GHz, 2C/2T, 7.84 GiB RAM, Linux 6.18.44-fc-v37 (#1 SMP PREEMPT_DYNAMIC @0)
- **Model:** `tiny-random-q8_0` (Q8_0, 21 MiB)
- **llama-server:** 0.1.0-dev (build 1, commit 4df29be)
- **Network:** REACHABLE - internet was available during this run
- **Backend sockets:** loopback only (127.0.0.1:56605)

## Metrics

| Metric | Value |
|---|---|
| Startup (spawn -> /health 200) | 0.15 s |
| RAM after load | 67 MiB |
| Peak RAM (os_peak_counter) | 69 MiB |
| Time to first token | 0.02 s |
| Prompt processing | 2876.98 tok/s |
| Generation | 974.61 tok/s |
| Tokens generated | 160 |

## Hardcoded prompt

> In two sentences, explain what a private, offline document question-answering assistant does.

## Response

```
Tom consult Play consult Play consult Play consult Play Tom consult Play──── consult Tom consult Tom consult Tom consult Tom consult Tom Tom Tom Tom Tom TomsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendsendespVIS Федерации div MC щ ChiesarareKEYespVIS Федерации щ Chiesarare щ Chiesarare щ
```

## Exact command

```
<llama.cpp>/bin/llama-server -m <models>/tiny-random-Q8_0.gguf --alias tiny-random-q8_0 --host 127.0.0.1 --port 56605 -c 4096 -t 2 -ngl 0 --offline --no-webui
```

Full record (config, system, network probes): `baseline.json`. Backend log: `./runs/logs/llama-server-20260922-233531-tiny-random-q8_0.log`.
