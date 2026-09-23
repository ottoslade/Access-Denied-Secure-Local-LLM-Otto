# Week 2 - Quantization performance matrix

> **SIMULATED RESULTS - stub backend.** These numbers come from the stub llama-server, not a real model. They prove the pipeline works end to end; re-run on the reference laptop with real GGUF files for real numbers.

- **When:** 2026-09-22T23:34:24-0400  |  **Prompt set:** 2026-09-22.1  |  **Runtime:** 0.2.0
- **Machine:** Intel(R) Xeon(R) Processor @ 2.10GHz, 2C/2T, 7.84 GiB RAM, Linux 6.18.44-fc-v37 (#1 SMP PREEMPT_DYNAMIC @0)
- **Settings:** 3 runs per prompt (median reported), max_tokens 96, ctx 4096, threads auto, GPU layers 0, temperature 0
- **Network during run:** reachable

## Results

| Model | Quant | File | Startup | Peak RAM | Gen tok/s (short) | RAG prompt tok/s | RAG time-to-first-token | RAG total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stub-1b-instruct | F16 | 1.86 GiB simulated | 2.18 s | 2.01 GiB | 7.48 | 69.52 | 15.761 s | 28.404 s |
| stub-1b-instruct | Q4_K_M | 583 MiB simulated | 0.79 s | 738 MiB | 24.47 | 222.45 | 4.925 s | 8.808 s |
| stub-1b-instruct | Q5_K_M | 678 MiB simulated | 0.92 s | 833 MiB | 21.0 | 187.21 | 5.85 s | 10.384 s |
| stub-1b-instruct | Q8_0 | 1013 MiB simulated | 1.18 s | 1.14 GiB | 14.11 | 124.11 | 8.824 s | 15.551 s |

## Compression vs. reference quant (same model)

| Model | Quant vs ref | File size | Peak RAM | Gen speed | RAG time-to-first-token | Startup |
|---|---|---:|---:|---:|---:|---:|
| stub-1b-instruct | Q4_K_M vs F16 | -69.4% | -64.2% | +227.1% | -68.8% | -63.6% |
| stub-1b-instruct | Q5_K_M vs F16 | -64.4% | -59.6% | +180.7% | -62.9% | -57.8% |
| stub-1b-instruct | Q8_0 vs F16 | -46.9% | -43.4% | +88.6% | -44.0% | -45.9% |

## Recommendation

Default for a 6 GB RAM budget: **`stub-1b-instruct-q8_0`** - highest-precision option (bits per weight, capped at Q6_K; ties go to lower RAM) that fits the 6 GB budget (peak 1.14 GiB) and is usable (>= 8 tok/s generation, <= 15 s to first token on the RAG prompt).

Answer quality is not measured here. Lower-bit quantizations are faster and smaller but can lose accuracy; the final pick should be confirmed against E5's grounding / citation eval set.

## How to read this

- **Startup**: process spawn until `/health` returns 200 (model loaded, ready to answer).
- **Peak RAM**: OS peak working set of llama-server (Windows `PeakWorkingSetSize`, Linux `VmHWM`). With mmap enabled, model pages count once they are touched.
- **Gen tok/s**: tokens generated per second as reported by llama.cpp `timings.predicted_per_second`.
- **RAG prompt tok/s / time-to-first-token**: how fast the model reads ~1.3k tokens of retrieved passages - on CPU this is usually the wait the user notices most in document Q&A.

Raw per-run data: `results.json`; spreadsheet-friendly: `results.csv`.
