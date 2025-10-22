"""Convert the bundled sample WAV file to μ-law 8KHz output."""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from audio_converter import AudioStreamConverter, MuLawStreamDecoder

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_INPUT = DATA_DIR / "sample_pcm16_24khz.wav"
DEFAULT_ULAW = DATA_DIR / "down_sample_8khz.ulaw"
DEFAULT_DECODED_8K = DATA_DIR / "down_sample_8khz.wav"
DEFAULT_UPSAMPLED = DATA_DIR / "up_sample_pcm16_24khz_decoded.wav"


def to_mono_pcm16_24khz(path: Path) -> np.ndarray:
    """Load WAV audio and return mono PCM16 24KHz samples.

    The bundled source is a 6-channel capture; we keep the first channel to
    preserve the original speaker while avoiding phase cancellation that
    occurs when averaging channels containing decorrelated ambience.
    """
    rate, data = wavfile.read(str(path))
    array = data.astype(np.float32, copy=False)

    if array.ndim > 1:
        array = array[:, 0]

    peak = float(np.max(np.abs(array))) or 1.0
    array = array / peak

    target_rate = 24000
    if rate != target_rate:
        gcd = math.gcd(rate, target_rate)
        up = target_rate // gcd
        down = rate // gcd
        array = signal.resample_poly(array, up, down).astype(np.float32, copy=False)

    pcm16 = np.clip(array, -1.0, 1.0)
    return (pcm16 * 32767).astype(np.int16)


def convert_wav_to_ulaw(input_wav: Path, output_ulaw: Path) -> None:
    """Convert the bundled WAV file to μ-law 8KHz using the converter."""
    pcm16 = to_mono_pcm16_24khz(input_wav)
    converter = AudioStreamConverter()

    with output_ulaw.open("wb") as outfile:
        for start in range(0, len(pcm16), converter.chunk_size):
            end = start + converter.chunk_size
            chunk = converter.convert_chunk(pcm16[start:end].tobytes())
            if chunk:
                outfile.write(chunk)

        final = converter.flush()
        if final:
            outfile.write(final)


def decode_ulaw_to_wav(
    input_ulaw: Path,
    output_wav: Path,
    *,
    input_rate: int = 8000,
    output_rate: int = 24000,
    apply_smoothing: bool = True,
) -> None:
    """Decode μ-law bytes to PCM16 and persist as a WAV file."""
    decoder = MuLawStreamDecoder(input_rate=input_rate, output_rate=output_rate)
    decoder.reset()

    pcm_chunks: list[np.ndarray] = []
    with input_ulaw.open("rb") as infile:
        chunk_size = decoder.chunk_size
        while chunk := infile.read(chunk_size):
            pcm = decoder.convert_chunk(chunk)
            if pcm:
                pcm_chunks.append(np.frombuffer(pcm, dtype=np.int16))

    final = decoder.flush()
    if final:
        pcm_chunks.append(np.frombuffer(final, dtype=np.int16))

    if not pcm_chunks:
        audio = np.empty(0, dtype=np.int16)
    else:
        audio = np.concatenate(pcm_chunks)

    if apply_smoothing and audio.size and output_rate > input_rate:
        nyquist = output_rate / 2.0
        cutoff = min(0.45 * input_rate, nyquist * 0.95)
        sos = signal.butter(4, cutoff, fs=output_rate, output="sos")
        smoothed = signal.sosfiltfilt(sos, audio.astype(np.float32))
        audio = np.clip(smoothed, -32768, 32767).astype(np.int16)

    with wave.open(str(output_wav), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(decoder.output_rate)
        wav_out.writeframes(audio.tobytes())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert bundled sample WAV to μ-law 8KHz via AudioStreamConverter."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input WAV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ULAW,
        help=f"Output μ-law file (default: {DEFAULT_ULAW})",
    )
    parser.add_argument(
        "--decoded-8k",
        type=Path,
        default=DEFAULT_DECODED_8K,
        help=f"Decoded 8KHz WAV output (default: {DEFAULT_DECODED_8K})",
    )
    parser.add_argument(
        "--decoded-24k",
        type=Path,
        default=DEFAULT_UPSAMPLED,
        help=f"Decoded 24KHz WAV output (default: {DEFAULT_UPSAMPLED})",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    convert_wav_to_ulaw(args.input, args.output)
    decode_ulaw_to_wav(
        args.output,
        args.decoded_8k,
        input_rate=8000,
        output_rate=8000,
    )
    decode_ulaw_to_wav(
        args.output,
        args.decoded_24k,
        input_rate=8000,
        output_rate=24000,
        apply_smoothing=True,
    )
    print(f"Converted {args.input} → {args.output}")
    print(f"Decoded μ-law to {args.decoded_8k}")
    print(f"Upsampled μ-law to {args.decoded_24k}")


if __name__ == "__main__":
    main()
