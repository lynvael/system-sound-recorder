"""Shared pytest environment for the test suite.

`pyaudiowpatch` ships Windows-only wheels (win32/win_amd64), so it is not
installed on the Linux dev box. The audio layer imports it lazily
(`app.audio.backend.get_backend()`), and the tests mock the backend anyway —
but `app.audio.capture` references module-level constants (`paFloat32`,
PortAudio error codes) through it, so we install a minimal fake module when
the real one is absent.

The fake's constants mirror the REAL PortAudio v19 values from the
PyAudioWPatch repo (`portaudio_v19/include/portaudio.h`), so error-code
mapping tests stay faithful to Windows behaviour. The `PyAudio` stub fails
loudly if a test actually tries to use the audio stack instead of mocking
`app.audio.backend.get_backend()`.
"""

import sys
import types

try:
    import pyaudiowpatch  # noqa: F401
except ImportError:
    _fake = types.ModuleType("pyaudiowpatch")

    # --- sample formats (PortAudio v19 PaSampleFormat — BIT FLAGS, not the
    # old v18/PyAudio enumeration: paFloat32 == 1, not 3) -------------------
    _fake.paFloat32 = 0x00000001
    _fake.paInt32 = 0x00000002
    _fake.paInt24 = 0x00000004
    _fake.paInt16 = 0x00000008
    _fake.paInt8 = 0x00000010
    _fake.paUInt8 = 0x00000020
    _fake.paCustomFormat = 0x00010000

    # --- error codes (PortAudio v19 PaErrorCode, see portaudio.h) ----------
    _fake.paNoError = 0
    _fake.paNotInitialized = -10000
    _fake.paUnanticipatedHostError = -9999
    _fake.paInvalidChannelCount = -9998
    _fake.paInvalidSampleRate = -9997
    _fake.paInvalidDevice = -9996
    _fake.paInvalidFlag = -9995
    _fake.paSampleFormatNotSupported = -9994
    _fake.paBadIODeviceCombination = -9993
    _fake.paInsufficientMemory = -9992
    _fake.paBufferTooBig = -9991
    _fake.paBufferTooSmall = -9990
    _fake.paNullCallback = -9989
    _fake.paBadStreamPtr = -9988
    _fake.paTimedOut = -9987
    _fake.paInternalError = -9986
    _fake.paDeviceUnavailable = -9985
    _fake.paIncompatibleHostApiSpecificStreamInfo = -9984
    _fake.paStreamIsStopped = -9983
    _fake.paStreamIsNotStopped = -9982
    _fake.paInputOverflowed = -9981
    _fake.paOutputUnderflowed = -9980
    _fake.paHostApiNotFound = -9979
    _fake.paInvalidHostApi = -9978
    _fake.paCanNotReadFromACallbackStream = -9977
    _fake.paCanNotWriteToACallbackStream = -9976
    _fake.paCanNotReadFromAnOutputOnlyStream = -9975
    _fake.paCanNotWriteToAnInputOnlyStream = -9974
    _fake.paIncompatibleStreamHostApi = -9973

    _fake.paFramesPerBufferUnspecified = -1

    class _FakePyAudio:
        """Stub that fails loudly if a test tries to use the real audio stack."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "pyaudiowpatch is not available on this platform; tests must "
                "mock app.audio.backend.get_backend() instead of using the "
                "real audio backend"
            )

    _fake.PyAudio = _FakePyAudio
    sys.modules["pyaudiowpatch"] = _fake
