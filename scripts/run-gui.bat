@echo off
REM Launches the Live Recorder GUI via uv, regardless of the current directory.
cd /d "%~dp0.."
uv run python -m app
