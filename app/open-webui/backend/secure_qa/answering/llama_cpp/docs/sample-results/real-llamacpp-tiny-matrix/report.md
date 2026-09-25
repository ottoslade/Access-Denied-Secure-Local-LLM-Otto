# Week 2 - Quantization performance matrix

- **When:** 2026-09-22T23:29:28-0400  |  **Prompt set:** 2026-09-22.1  |  **Runtime:** 0.2.0
- **Machine:** Intel(R) Xeon(R) Processor @ 2.10GHz, 2C/2T, 7.84 GiB RAM, Linux 6.18.44-fc-v37 (#1 SMP PREEMPT_DYNAMIC @0)
- **Settings:** 3 runs per prompt (median reported), max_tokens 64, ctx 4096, threads auto, GPU layers 0, temperature 0
- **Network during run:** reachable

## Results

| Model | Quant | File | Startup | Peak RAM | Gen tok/s (short) | RAG prompt tok/s | RAG time-to-first-token | RAG total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tiny-random | F16 | 38 MiB | 0.16 s | 101 MiB | 817.9 | 11876.62 | 0.099 s | 0.269 s |
| tiny-random | F32 | 76 MiB | 0.15 s | 133 MiB | 611.89 | 10806.84 | 0.107 s | 0.301 s |
| tiny-random | Q4_K_M | 13 MiB | 0.15 s | 73 MiB | 1213.92 | 14891.87 | 0.076 s | 0.228 s |
| tiny-random | Q8_0 | 21 MiB | 0.15 s | 83 MiB | 1278.98 | 17269.83 | 0.071 s | 0.217 s |

## Compression vs. reference quant (same model)

| Model | Quant vs ref | File size | Peak RAM | Gen speed | RAG time-to-first-token | Startup |
|---|---|---:|---:|---:|---:|---:|
| tiny-random | F16 vs F32 | -49.5% | -23.8% | +33.7% | -7.5% | +1.9% |
| tiny-random | Q4_K_M vs F32 | -82.3% | -45.4% | +98.4% | -29.0% | -0.6% |
| tiny-random | Q8_0 vs F32 | -72.8% | -37.9% | +109.0% | -33.6% | -0.6% |

## Recommendation

Default for a 6 GB RAM budget: **`tiny-random-q8_0`** - highest-precision option (bits per weight, capped at Q6_K; ties go to lower RAM) that fits the 6 GB budget (peak 83 MiB) and is usable (>= 8 tok/s generation, <= 15 s to first token on the RAG prompt).

Answer quality is not measured here. Lower-bit quantizations are faster and smaller but can lose accuracy; the final pick should be confirmed against E5's grounding / citation eval set.

## How to read this

- **Startup**: process spawn until `/health` returns 200 (model loaded, ready to answer).
- **Peak RAM**: OS peak working set of llama-server (Windows `PeakWorkingSetSize`, Linux `VmHWM`). With mmap enabled, model pages count once they are touched.
- **Gen tok/s**: tokens generated per second as reported by llama.cpp `timings.predicted_per_second`.
- **RAG prompt tok/s / time-to-first-token**: how fast the model reads ~1.3k tokens of retrieved passages - on CPU this is usually the wait the user notices most in document Q&A.

Raw per-run data: `results.json`; spreadsheet-friendly: `results.csv`.
