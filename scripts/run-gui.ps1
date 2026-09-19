# Launches the Live Recorder GUI via uv, regardless of the current directory.
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
uv run python -m app
