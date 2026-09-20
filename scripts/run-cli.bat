@echo off
REM Forwards all arguments to the Live Recorder CLI via uv, e.g. run-cli.bat transcribe file.wav
cd /d "%~dp0.."
uv run python -m app.cli %*
