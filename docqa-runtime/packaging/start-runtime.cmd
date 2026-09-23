@echo off
rem Double-click to start the DocQA runtime. Close this window (or press Ctrl+C) to stop it.
cd /d "%~dp0"
docqa-runtime.exe up %*
if errorlevel 1 (
  echo.
  echo The runtime stopped with an error - read the message above.
  pause
)
