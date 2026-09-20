# Forwards all arguments to the Live Recorder CLI via uv, e.g. ./run-cli.ps1 transcribe file.wav
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
uv run python -m app.cli @args
