"""
Audio Converter Plugin: PCM16 24KHz to μ-law 8KHz
Converts streaming audio from Gemini Live API format to Twilio-compatible format.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable, Iterator
from typing import Final, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

Int16Array = NDArray[np.int16]

PCM_BYTES_PER_SAMPLE: Final[int] = np.dtype(np.int16).itemsize
ULAW_BYTES_PER_SAMPLE: Final[int] = 1
MIN_BUFFER_MS: Final[int] = 10
PathLike: TypeAlias = Union[str, os.PathLike[str]]

ULAW_BIAS: Final[int] = 0x84


class AudioStreamConverter:
    """
    Converts PCM16 24KHz audio stream to μ-law 8KHz format in real-time.

    This converter is designed to work with streaming audio data, processing
    chunks as they arrive from the Gemini Live API and outputting converted
    chunks suitable for services like Twilio.
    """

    __slots__ = (
        "input_rate",
        "output_rate",
        "chunk_size",
        "resample_ratio",
        "residual_buffer",
        "_byte_buffer",
        "gcd",
        "up_factor",
        "down_factor",
        "min_chunk_size",
        "_resample_filter",
        "_ulaw_lut",
        "_tail_input_size",
        "_tail_output_size",
        "_chunk_bytes",
    )

    def __init__(
        self,
        input_rate: int = 24000,
        output_rate: int = 8000,
        chunk_size: int = 4800,
    ) -> None:
        """
        Initialize the audio stream converter.

        Args:
            input_rate: Input sample rate in Hz (default: 24000)
            output_rate: Output sample rate in Hz (default: 8000)
            chunk_size: Size of input chunks in samples (default: 4800 = 200ms at 24KHz)
        """
        if input_rate <= 0:
            raise ValueError("input_rate must be a positive integer")
        if output_rate <= 0:
            raise ValueError("output_rate must be a positive integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        self.input_rate = input_rate
        self.output_rate = output_rate
        self.chunk_size = chunk_size
        self.resample_ratio = output_rate / input_rate

        # Buffer for handling partial samples across chunks
        self.residual_buffer = np.empty(0, dtype=np.int16)
        self._byte_buffer: bytes = b""

        # Calculate resampling parameters
        self.gcd = np.gcd(input_rate, output_rate)
        self.up_factor = output_rate // self.gcd
        self.down_factor = input_rate // self.gcd
        min_buffer_samples = max(1, math.ceil((self.input_rate * MIN_BUFFER_MS) / 1000))
        self.min_chunk_size = max(1, min(self.chunk_size, min_buffer_samples))
        self._chunk_bytes = self.chunk_size * PCM_BYTES_PER_SAMPLE

        # Pre-compute helpers used during conversion for better throughput
        self._resample_filter = self._design_resample_filter()
        self._ulaw_lut = self._build_ulaw_lookup()

        # Tail sizes control how much context we retain between chunks
        self._tail_input_size = len(self._resample_filter)
        self._tail_output_size = (
            int(math.ceil(self._tail_input_size * (self.output_rate / self.input_rate)))
            + self.down_factor
        )

    def __repr__(self) -> str:  # pragma: no cover - for debugging convenience
        return (
            "AudioStreamConverter("
            f"input_rate={self.input_rate}, "
            f"output_rate={self.output_rate}, "
            f"chunk_size={self.chunk_size})"
        )

    def _pcm16_to_numpy(self, pcm_data: bytes) -> Int16Array:
        """
        Convert PCM16 bytes to numpy array.

        Args:
            pcm_data: Raw PCM16 audio bytes (little-endian)

        Returns:
            NumPy array of int16 samples
        """
        return cast(Int16Array, np.frombuffer(pcm_data, dtype=np.int16))

    def _design_resample_filter(self) -> NDArray[np.float32]:
        """
        Build and cache the FIR filter coefficients used by resample_poly.
        """
        max_rate = max(self.up_factor, self.down_factor)
        cutoff = 1.0 / max_rate
        half_len = 10 * max_rate
        taps = signal.firwin(2 * half_len + 1, cutoff, window=("kaiser", 5.0))
        taps *= self.up_factor
        return cast(NDArray[np.float32], taps.astype(np.float32, copy=False))

    @staticmethod
    def _build_ulaw_lookup() -> NDArray[np.uint8]:
        """
        Generate a lookup table mapping int16 PCM samples to μ-law bytes.
        """
        bias = 0x84
        clip = 32635

        values = np.arange(-32768, 32768, dtype=np.int32)
        sign = (values < 0).astype(np.uint8)
        magnitude = np.abs(values)
        magnitude = np.clip(magnitude, 0, clip)
        magnitude = magnitude + bias

        exponent = np.zeros_like(magnitude, dtype=np.uint8)
        for i in range(7):
            mask = 1 << (i + 8)
            exponent = np.where(magnitude >= mask, i + 1, exponent)

        mantissa = (magnitude >> (exponent + 3)) & 0x0F
        ulaw = ((sign << 7) | (exponent << 4) | mantissa) ^ 0xFF
        return ulaw.astype(np.uint8)

    def _resample_chunk(self, samples: Int16Array) -> Int16Array:
        """
        Resample audio chunk from input_rate to output_rate.

        Uses scipy's resample_poly for high-quality resampling with anti-aliasing.

        Args:
            samples: Input audio samples as int16 numpy array

        Returns:
            Resampled audio as int16 numpy array
        """
        # Convert to float for resampling
        samples_float = samples.astype(np.float32)

        # Perform resampling with polyphase filtering (high quality, efficient)
        resampled = signal.resample_poly(
            samples_float,
            self.up_factor,
            self.down_factor,
            window=self._resample_filter,
        )

        # Clip and convert back to int16
        resampled = np.clip(resampled, -32768, 32767)
        return cast(Int16Array, resampled.astype(np.int16))

    def _pcm16_to_ulaw(self, pcm_data: Int16Array) -> bytes:
        """
        Convert PCM16 samples to μ-law encoded bytes.

        Implements ITU-T G.711 μ-law compression algorithm.

        Args:
            pcm_data: PCM16 audio samples as int16 numpy array

        Returns:
            μ-law encoded audio bytes
        """
        if pcm_data.size == 0:
            return b""

        indices = pcm_data.astype(np.int32) + 32768
        lut_slice = cast(NDArray[np.uint8], self._ulaw_lut[indices])
        return lut_slice.tobytes()

    def convert_chunk(self, pcm16_data: bytes) -> bytes | None:
        """
        Convert a single chunk of PCM16 24KHz audio to μ-law 8KHz.

        This method handles streaming conversion, maintaining internal state
        for smooth transitions between chunks.

        Args:
            pcm16_data: Raw PCM16 audio bytes at 24KHz

        Returns:
            μ-law encoded audio bytes at 8KHz, or None if not enough data
        """
        if not pcm16_data and not self.residual_buffer.size:
            return None

        if self._byte_buffer:
            pcm16_data = self._byte_buffer + pcm16_data
            self._byte_buffer = b""

        if len(pcm16_data) % 2 == 1:
            # Keep the trailing byte so we always convert full samples
            self._byte_buffer = pcm16_data[-1:]
            pcm16_data = pcm16_data[:-1]

        if not pcm16_data:
            return None

        # Convert input bytes to numpy array
        samples = self._pcm16_to_numpy(pcm16_data)

        # Combine with residual from previous chunk
        if len(self.residual_buffer) > 0:
            samples = np.concatenate([self.residual_buffer, samples])

        if len(samples) < self.min_chunk_size:
            # Not enough data yet, keep everything for the next call
            self.residual_buffer = samples.copy()
            return None

        # Resample to 8KHz
        resampled = self._resample_chunk(samples)

        # Determine how much to retain for the next chunk to keep filter state
        tail_input_len = min(len(samples), self._tail_input_size)
        tail_output_len = min(len(resampled), self._tail_output_size)
        emit_len = len(resampled) - tail_output_len

        if emit_len <= 0:
            # Still waiting for enough data to produce stable output
            self.residual_buffer = samples.copy()
            return None

        emit_samples = resampled[:emit_len]
        if tail_input_len == 0:
            self.residual_buffer = np.empty(0, dtype=np.int16)
        else:
            self.residual_buffer = samples[-tail_input_len:].copy()

        # Convert to μ-law
        ulaw_data = self._pcm16_to_ulaw(emit_samples.astype(np.int16, copy=False))

        return ulaw_data

    def convert_stream(self, audio_stream: Iterable[bytes]) -> Iterator[bytes]:
        """
        Convert a stream of PCM16 24KHz audio chunks to μ-law 8KHz.

        This is a generator function that yields converted chunks as they
        are processed, suitable for real-time streaming applications.

        Args:
            audio_stream: Iterator yielding PCM16 audio byte chunks at 24KHz

        Yields:
            μ-law encoded audio byte chunks at 8KHz

        Example:
            >>> converter = AudioStreamConverter()
            >>> for ulaw_chunk in converter.convert_stream(gemini_audio_stream):
            ...     # Send to Twilio or other service
            ...     twilio_stream.send(ulaw_chunk)
        """
        for chunk in audio_stream:
            converted = self.convert_chunk(chunk)
            if converted is not None:
                yield converted

    def flush(self) -> bytes | None:
        """
        Flush any remaining audio data in the buffer.

        Call this at the end of a stream to process any residual samples.

        Returns:
            Final μ-law encoded audio bytes, or None if buffer is empty
        """
        if self._byte_buffer:
            # Discard incomplete sample at end of stream
            self._byte_buffer = b""

        if len(self.residual_buffer) == 0:
            return None

        # Process remaining samples
        resampled = self._resample_chunk(self.residual_buffer)
        ulaw_data = self._pcm16_to_ulaw(resampled)

        # Clear buffer
        self.residual_buffer = np.empty(0, dtype=np.int16)

        return ulaw_data

    def reset(self) -> None:
        """
        Reset the converter state, clearing all buffers.

        Useful when starting a new audio stream.
        """
        self.residual_buffer = np.empty(0, dtype=np.int16)
        self._byte_buffer = b""


class MuLawStreamDecoder:
    """
    Converts μ-law 8KHz audio stream to PCM16 24KHz format.

    Useful for processing audio received from telephony providers back into
    higher-fidelity PCM16 audio.
    """

    __slots__ = (
        "input_rate",
        "output_rate",
        "chunk_size",
        "resample_ratio",
        "residual_buffer",
        "_decode_lut",
        "_needs_resample",
        "gcd",
        "up_factor",
        "down_factor",
        "min_chunk_size",
        "_resample_filter",
        "_tail_input_size",
        "_tail_output_size",
        "_chunk_bytes",
    )

    def __init__(
        self,
        input_rate: int = 8000,
        output_rate: int = 24000,
        chunk_size: int = 1600,
    ) -> None:
        if input_rate <= 0:
            raise ValueError("input_rate must be a positive integer")
        if output_rate <= 0:
            raise ValueError("output_rate must be a positive integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        self.input_rate = input_rate
        self.output_rate = output_rate
        self.chunk_size = chunk_size
        self.resample_ratio = output_rate / input_rate

        self.residual_buffer = np.empty(0, dtype=np.int16)

        self.gcd = math.gcd(input_rate, output_rate)
        self.up_factor = output_rate // self.gcd
        self.down_factor = input_rate // self.gcd
        min_buffer_samples = max(1, math.ceil((self.input_rate * MIN_BUFFER_MS) / 1000))
        self.min_chunk_size = max(1, min(self.chunk_size, min_buffer_samples))
        self._chunk_bytes = self.chunk_size * ULAW_BYTES_PER_SAMPLE

        self._decode_lut = self._build_decode_lookup()
        self._needs_resample = not (self.up_factor == 1 and self.down_factor == 1)

        if self._needs_resample:
            self._resample_filter = self._design_resample_filter()
            self._tail_input_size = len(self._resample_filter)
            self._tail_output_size = (
                int(
                    math.ceil(
                        self._tail_input_size
                        * (self.output_rate / self.input_rate)
                    )
                )
                + self.down_factor
            )
        else:
            self._resample_filter = np.array([1.0], dtype=np.float32)
            self._tail_input_size = 0
            self._tail_output_size = 0

    def __repr__(self) -> str:  # pragma: no cover - helper for debugging
        return (
            "MuLawStreamDecoder("
            f"input_rate={self.input_rate}, "
            f"output_rate={self.output_rate}, "
            f"chunk_size={self.chunk_size})"
        )

    def _design_resample_filter(self) -> NDArray[np.float32]:
        max_rate = max(self.up_factor, self.down_factor)
        if max_rate <= 1:
            return np.array([1.0], dtype=np.float32)
        cutoff = 1.0 / max_rate
        half_len = 10 * max_rate
        taps = signal.firwin(2 * half_len + 1, cutoff, window=("kaiser", 5.0))
        taps *= self.up_factor
        return cast(NDArray[np.float32], taps.astype(np.float32, copy=False))

    @staticmethod
    def _build_decode_lookup() -> Int16Array:
        mu = np.arange(256, dtype=np.uint8)
        mu = np.bitwise_xor(mu, 0xFF)

        sign = mu & 0x80
        exponent = (mu >> 4) & 0x07
        mantissa = mu & 0x0F

        sample = mantissa.astype(np.int32) << 3
        sample += ULAW_BIAS
        sample <<= exponent
        sample -= ULAW_BIAS
        sample = np.where(sign != 0, -sample, sample)
        sample = np.clip(sample, -32768, 32767)
        return sample.astype(np.int16)

    def _decode_ulaw(self, data: bytes) -> Int16Array:
        if not data:
            return np.empty(0, dtype=np.int16)
        values = np.frombuffer(data, dtype=np.uint8)
        return self._decode_lut[values]

    def _resample_chunk(self, samples: Int16Array) -> Int16Array:
        if samples.size == 0:
            return samples
        if not self._needs_resample:
            return samples
        samples_float = samples.astype(np.float32)
        resampled = signal.resample_poly(
            samples_float,
            self.up_factor,
            self.down_factor,
            window=self._resample_filter,
        )
        resampled = np.clip(resampled, -32768, 32767)
        return cast(Int16Array, resampled.astype(np.int16))

    def convert_chunk(self, ulaw_data: bytes) -> bytes | None:
        if not ulaw_data and not self.residual_buffer.size:
            return None

        samples = self._decode_ulaw(ulaw_data)
        if self.residual_buffer.size:
            samples = np.concatenate([self.residual_buffer, samples])

        if len(samples) < self.min_chunk_size:
            self.residual_buffer = samples.copy()
            return None

        resampled = self._resample_chunk(samples)

        tail_input_len = min(len(samples), self._tail_input_size)
        tail_output_len = min(len(resampled), self._tail_output_size)
        emit_len = len(resampled) - tail_output_len

        if emit_len <= 0:
            self.residual_buffer = samples.copy()
            return None

        emit_samples = resampled[:emit_len]
        if tail_input_len == 0:
            self.residual_buffer = np.empty(0, dtype=np.int16)
        else:
            self.residual_buffer = samples[-tail_input_len:].copy()
        return emit_samples.tobytes()

    def convert_stream(self, audio_stream: Iterable[bytes]) -> Iterator[bytes]:
        for chunk in audio_stream:
            converted = self.convert_chunk(chunk)
            if converted is not None:
                yield converted

    def flush(self) -> bytes | None:
        if not self.residual_buffer.size:
            return None

        resampled = self._resample_chunk(self.residual_buffer)
        self.residual_buffer = np.empty(0, dtype=np.int16)
        return resampled.tobytes()

    def reset(self) -> None:
        self.residual_buffer = np.empty(0, dtype=np.int16)


def convert_file(
    input_path: PathLike,
    output_path: PathLike,
    *,
    chunk_samples: int | None = None,
    converter: AudioStreamConverter | None = None,
) -> None:
    """
    Utility function to convert an entire audio file.

    Args:
        input_path: Path to input PCM16 24KHz file
        output_path: Path to output μ-law 8KHz file
        chunk_samples: Optional custom chunk size in samples
        converter: Optional pre-configured converter instance. The converter
            will be reset before use to ensure a clean state.
    """
    converter = converter or AudioStreamConverter()
    converter.reset()

    if chunk_samples is None:
        chunk_samples = converter.chunk_size
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be a positive integer")

    chunk_bytes = chunk_samples * PCM_BYTES_PER_SAMPLE

    logger.debug(
        "Converting PCM16 to μ-law: input=%s output=%s chunk_samples=%s",
        input_path,
        output_path,
        chunk_samples,
    )

    with open(input_path, "rb") as infile, open(output_path, "wb") as outfile:
        while True:
            chunk = infile.read(chunk_bytes)
            if not chunk:
                break

            converted = converter.convert_chunk(chunk)
            if converted:
                outfile.write(converted)

        # Flush remaining data
        final_chunk = converter.flush()
        if final_chunk:
            outfile.write(final_chunk)


def convert_ulaw_file(
    input_path: PathLike,
    output_path: PathLike,
    *,
    chunk_samples: int | None = None,
    decoder: MuLawStreamDecoder | None = None,
) -> None:
    """
    Convert an entire μ-law 8KHz audio file to PCM16 24KHz.

    Args:
        input_path: Path to input μ-law file
        output_path: Path to output PCM16 file
        chunk_samples: Optional chunk size in μ-law samples
        decoder: Optional decoder instance to reuse
    """
    decoder = decoder or MuLawStreamDecoder()
    decoder.reset()

    if chunk_samples is None:
        chunk_samples = decoder.chunk_size
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be a positive integer")

    chunk_bytes = chunk_samples * ULAW_BYTES_PER_SAMPLE

    logger.debug(
        "Converting μ-law to PCM16: input=%s output=%s chunk_samples=%s",
        input_path,
        output_path,
        chunk_samples,
    )

    with open(input_path, "rb") as infile, open(output_path, "wb") as outfile:
        while True:
            chunk = infile.read(chunk_bytes)
            if not chunk:
                break

            converted = decoder.convert_chunk(chunk)
            if converted:
                outfile.write(converted)

        final_chunk = decoder.flush()
        if final_chunk:
            outfile.write(final_chunk)
