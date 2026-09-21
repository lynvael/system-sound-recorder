"""Shared pytest environment guards for the test suite."""

import sys

# Defensive guard against argv access under bare `pytest` / `python -c`
# invocation: `soundcard` (imported transitively by app.pipeline.session)
# reads sys.argv[1] when inferring its PulseAudio program name — at runtime,
# inside `_PulseAudio.__init__`, not at import time — which raises IndexError
# when argv has fewer than 2 entries. pytest normally passes the test path as
# argv[1]; guard the no-args case so collection never dies inside the
# capture layer.
if len(sys.argv) < 2:
    sys.argv.append("pytest")
