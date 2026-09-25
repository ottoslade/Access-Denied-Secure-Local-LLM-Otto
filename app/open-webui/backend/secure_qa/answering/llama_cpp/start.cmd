@echo off
rem Start the DocQA runtime from source and chat with it. Double-click, or run from a terminal.
rem Once the model is loaded, type in this window; "exit" (or Ctrl+C) stops the runtime.
rem First run creates .venv and installs the package; later runs start straight away.
rem Uses llama.cpp when bin\llama-server.exe (or llama-server on PATH) and a models\*.gguf exist,
rem otherwise falls back to the stub backend (simulated answers, no download needed).
rem   start.cmd [flags]                 flags go to "docqa-runtime up", e.g. --model <id> --port 9090
rem   start.cmd --server-only [flags]   API server only, no chat prompt (e.g. for the desktop app)
setlocal
cd /d "%~dp0"
set "DQ=.venv\Scripts\docqa-runtime.exe"

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python environment in .venv ...
  py -3 -m venv .venv 2>nul || python -m venv .venv || goto :fail
)
if not exist "%DQ%" (
  echo Installing docqa-runtime into .venv ...
  ".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -e . || goto :fail
)

set "HAVE_BIN="
set "HAVE_MODEL="
if exist "bin\llama-server.exe" set "HAVE_BIN=1"
where llama-server >nul 2>nul && set "HAVE_BIN=1"
for %%f in (models\*.gguf) do set "HAVE_MODEL=1"

set "EXTRA="
if defined HAVE_BIN if defined HAVE_MODEL goto :run
echo.
echo llama.cpp or a .gguf model was not found - starting the STUB backend (simulated answers).
echo For a real model run scripts\windows\fetch_runtime.ps1 and scripts\windows\fetch_models.ps1
echo.
"%DQ%" stub-models --out runs\stub-models >nul || goto :fail
set "EXTRA=--server-bin stub --models-dir runs\stub-models"

:run
set "MODE=--chat"
if /i "%~1"=="--server-only" set "MODE="
set "ARGS=%*"
if not defined MODE set "ARGS=%ARGS:*--server-only=%"
"%DQ%" up %MODE% %EXTRA% %ARGS%
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo The runtime stopped with an error - read the message above.
pause
exit /b 1
