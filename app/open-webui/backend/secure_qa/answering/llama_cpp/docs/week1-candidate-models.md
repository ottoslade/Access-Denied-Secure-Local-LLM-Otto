# Week 1: Candidate model matrix and baseline protocol

**Target:** the Windows reference laptop, 8–16 GB RAM, CPU only, air-gapped. The runtime is llama.cpp (`llama-server`)
with text-only GGUF models.

## Selection criteria

1. **Fits in memory with headroom.** Peak RAM with 4k context must stay under about 6 GB, so the OS, the app and the
   vector store still fit on an 8 GB machine.
2. **Grounded answering.** The model must follow "answer only from the passages and cite them". That rules out most
   models under 1.5B for production use.
3. **Usable context.** At least 8k tokens of context, so several retrieved chunks fit alongside the question.
4. **Licence.** Must be usable in the product without a commercial-use problem.
5. **Speed on CPU.** At least 8 tokens/s generation, and a reasonable time to first token for about 1.3k tokens of
   retrieved context. `docqa-runtime bench` measures both.

## Shortlist

The machine-readable version is `models/shortlist.json`, and `fetch_models.ps1` downloads from it. Sizes are
approximate Q4_K_M file sizes.

| Candidate | Params | Q4_K_M | Context | Licence | Role |
|---|---:|---:|---:|---|---|
| **Qwen3-4B-Instruct-2507** | 4.0B | ~2.5 GB | 256k | Apache-2.0 | **Primary.** Best instruction following in class; non-thinking variant |
| Llama-3.2-3B-Instruct | 3.2B | ~2.0 GB | 128k | Llama 3.2 Community | Fallback, widely used baseline |
| Phi-3.5-mini-instruct | 3.8B | ~2.4 GB | 128k | MIT | Fallback, strong reading comprehension |
| Qwen2.5-1.5B-Instruct | 1.5B | ~1.0 GB | 32k | Apache-2.0 | Low-spec / smoke-test model |
| SmolLM2-1.7B-Instruct | 1.7B | ~1.1 GB | 8k | Apache-2.0 | Fastest. Short context limits it to small docs |

**Not shortlisted:** 7B+ models (Q4 is about 4.5 GB plus KV cache, which is too tight on 8 GB alongside the app);
reasoning or "thinking" models (slow on CPU, and their output needs stripping); vision-language models (not needed,
larger); Qwen2.5-3B (its research licence restricts commercial use).

**Before relying on this list:** the model landscape moves fast. Check Hugging Face for newer small instruct models
and confirm each licence with whoever owns legal sign-off. The benchmark takes any GGUF you drop into `models\`,
so adding a candidate costs nothing.

## Baseline protocol (Definition of Done: Week 1)

On the reference laptop:

1. `scripts\windows\fetch_runtime.ps1` and `scripts\windows\fetch_models.ps1 -Set week2-small`, on a connected
   machine. Copy the folder over.
2. Disable networking: airplane mode, or `scripts\windows\block-network.ps1` as Administrator.
3. `docqa-runtime doctor`. Every line should be OK, including "no outbound network (offline)".
4. `docqa-runtime baseline --require-offline`. This refuses to run if the internet is reachable. It then:
   * starts llama-server with the exact recorded command (`-m … --host 127.0.0.1 --offline …`);
   * sends the fixed prompt *"In two sentences, explain what a private, offline document question-answering
     assistant does."* and streams the answer to the console;
   * writes `runs\baseline-<timestamp>\baseline.md` and `baseline.json` with startup time, RAM after load, **peak RAM**
     (Windows `PeakWorkingSetSize`), **tokens/s** (prompt and generation, from llama.cpp's own timings),
     time to first token, the full config, machine specs, network probe results, and a socket audit showing
     llama-server listens on loopback only.

The Week 1 record from this workspace (stub and a tiny real llama.cpp model) is in `docs/sample-results/`.
It shows the format but is **not** reference-laptop data.
