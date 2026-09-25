DocQA Runtime - portable bundle
================================

Folder layout
  docqa-runtime.exe     the launcher / local API
  runtime.toml          settings (port, model, context size, ...)
  bin\                  llama.cpp for Windows (llama-server.exe + DLLs)
  models\               .gguf model files
  runs\                 logs, baseline records, benchmark reports
  scripts\              block-network.ps1 (firewall), demo.ps1 (Friday demo)

Start
  Double-click start-runtime.cmd, or in a terminal:   docqa-runtime up
  It prints the endpoints when ready:
    GET  http://127.0.0.1:8080/health
    GET  http://127.0.0.1:8080/v1/models
    POST http://127.0.0.1:8080/v1/chat/completions

Other commands
  docqa-runtime doctor                check setup (binary, model, RAM, port, network)
  docqa-runtime models                list models found in models\
  docqa-runtime chat "question"       ask the running runtime
  docqa-runtime baseline --require-offline
                                      Week 1 record: hardcoded prompt, startup, peak RAM, tokens/s
  docqa-runtime bench                 compare every model/quantization in models\
  docqa-runtime up --server-bin stub  dry run without llama.cpp (after: docqa-runtime stub-models)

Offline
  The runtime only listens on 127.0.0.1, passes --offline to llama.cpp and removes download settings
  from its environment. For extra assurance run scripts\block-network.ps1 as Administrator, or use
  airplane mode during the demo.
