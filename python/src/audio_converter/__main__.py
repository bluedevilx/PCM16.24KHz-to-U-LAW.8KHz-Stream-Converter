"""Command line entry point for the audio converter."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .converter import (
    AudioStreamConverter,
    MuLawStreamDecoder,
    convert_file,
    convert_ulaw_file,
)

LOG_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

DEFAULT_FORWARD_CHUNK = 4800
DEFAULT_REVERSE_CHUNK = 1600


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert between PCM16 24KHz and μ-law 8KHz audio."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to PCM16 24KHz input file.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to output μ-law file.",
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=None,
        help=(
            "Chunk size to read (number of PCM samples per chunk). "
            "Defaults to the converter chunk size."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=("pcm-to-ulaw", "ulaw-to-pcm"),
        default="pcm-to-ulaw",
        help="Conversion direction (default: pcm-to-ulaw).",
    )
    parser.add_argument(
        "--input-rate",
        type=int,
        default=None,
        help="Input sample rate in Hz (default depends on direction).",
    )
    parser.add_argument(
        "--output-rate",
        type=int,
        default=None,
        help="Output sample rate in Hz (default depends on direction).",
    )
    parser.add_argument(
        "--log-level",
        choices=sorted(LOG_LEVELS),
        default="warning",
        help="Logging verbosity (default: warning).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"audio-stream-converter {__version__}",
    )
    return parser


def configure_logging(level: str) -> None:
    """Initialise logging for the CLI."""
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=LOG_LEVELS.get(level, logging.WARNING),
    )


def run_cli(args: argparse.Namespace) -> int:
    """Run the conversion based on parsed arguments."""
    configure_logging(args.log_level)
    if args.direction == "pcm-to-ulaw":
        input_rate = args.input_rate or 24000
        output_rate = args.output_rate or 8000
        chunk_samples = args.chunk_samples or DEFAULT_FORWARD_CHUNK
        converter = AudioStreamConverter(
            input_rate=input_rate,
            output_rate=output_rate,
            chunk_size=chunk_samples,
        )
        convert_file(
            args.input_path,
            args.output_path,
            chunk_samples=chunk_samples,
            converter=converter,
        )
        logging.getLogger(__name__).info(
            "Converted PCM16 %sHz → μ-law %sHz: %s → %s",
            input_rate,
            output_rate,
            args.input_path,
            args.output_path,
        )
    else:
        input_rate = args.input_rate or 8000
        output_rate = args.output_rate or 24000
        chunk_samples = args.chunk_samples or DEFAULT_REVERSE_CHUNK
        decoder = MuLawStreamDecoder(
            input_rate=input_rate,
            output_rate=output_rate,
            chunk_size=chunk_samples,
        )
        convert_ulaw_file(
            args.input_path,
            args.output_path,
            chunk_samples=chunk_samples,
            decoder=decoder,
        )
        logging.getLogger(__name__).info(
            "Converted μ-law %sHz → PCM16 %sHz: %s → %s",
            input_rate,
            output_rate,
            args.input_path,
            args.output_path,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
