"""Public package interface for the audio stream converter."""

from .converter import (
    AudioStreamConverter,
    MuLawStreamDecoder,
    convert_file,
    convert_ulaw_file,
)

__all__ = [
    "AudioStreamConverter",
    "MuLawStreamDecoder",
    "convert_file",
    "convert_ulaw_file",
    "__version__",
]
__version__ = "0.1.0"
