# Sample results: format only, not reference-laptop data

These were produced in a Linux build workspace (2-core Xeon, 8 GB RAM) to show what the Week 1 and Week 2 outputs look
like. **None of these numbers describe the reference laptop or a real model.**

| Folder | Backend | Model | What it proves |
|---|---|---|---|
| `stub-baseline/` | stub (simulated) | simulated 1B at Q4_K_M | Week 1 record format |
| `stub-quant-matrix/` | stub (simulated) | simulated 1B at F16 / Q8_0 / Q5_K_M / Q4_K_M | Week 2 report, comparison table and recommendation |
| `real-llamacpp-tiny-baseline/` | **real llama-server** (llama.cpp built from source) | 19M-parameter random-weight test model | The launcher, `--offline`, the backend key and the metrics all work against the genuine engine. The answer text is gibberish by design |
| `real-llamacpp-tiny-matrix/` | **real llama-server** | same test model at F32 / F16 / Q8_0 / Q4_K_M | Real quantized files are loaded and measured; RAM falls with compression as expected |

The real numbers come from running `docqa-runtime baseline --require-offline` and `docqa-runtime bench` on the
reference laptop with the shortlisted models.
