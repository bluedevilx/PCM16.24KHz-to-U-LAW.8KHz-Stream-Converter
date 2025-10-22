"""Test suite for the audio stream converter."""

from __future__ import annotations

import math
import wave
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

from audio_converter import (
    AudioStreamConverter,
    MuLawStreamDecoder,
    convert_file,
    convert_ulaw_file,
)


def sine_wave(
    samples: int,
    *,
    sample_rate: int,
    frequency: float = 1000.0,
    amplitude: int = 16000,
) -> np.ndarray:
    """Generate a sine wave of the given length as int16 samples."""
    t = np.linspace(0, samples / sample_rate, samples, endpoint=False, dtype=np.float32)
    waveform = amplitude * np.sin(2.0 * np.pi * frequency * t)
    return waveform.astype(np.int16, copy=False)


@pytest.fixture()
def converter() -> AudioStreamConverter:
    """Provide a fresh converter for each test."""
    return AudioStreamConverter()


def run_conversion(chunks: Iterable[bytes]) -> bytes:
    """Utility to convert a series of chunks and return the combined output."""
    conv = AudioStreamConverter()
    outputs: list[bytes] = []
    for chunk in chunks:
        maybe = conv.convert_chunk(chunk)
        if maybe:
            outputs.append(maybe)
    final = conv.flush()
    if final:
        outputs.append(final)
    return b"".join(outputs)


def load_sample_pcm16(limit_seconds: float = 1.0) -> np.ndarray:
    """Load the bundled sample WAV and convert to PCM16 24KHz mono."""
    sample_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "data"
        / "sample_pcm16_24khz.wav"
    )
    rate, data = wavfile.read(str(sample_path))

    array = data.astype(np.float32, copy=False)
    if array.ndim > 1:
        array = array[:, 0]

    if limit_seconds is not None:
        max_samples = int(rate * limit_seconds)
        array = array[:max_samples]

    peak = float(np.max(np.abs(array))) or 1.0
    array = array / peak

    if rate != 24000:
        gcd = math.gcd(rate, 24000)
        up = 24000 // gcd
        down = rate // gcd
        array = signal.resample_poly(array, up, down).astype(np.float32, copy=False)

    pcm16 = np.clip(array, -1.0, 1.0)
    return (pcm16 * 32767).astype(np.int16)


def test_basic_conversion_emits_expected_length(
    converter: AudioStreamConverter,
) -> None:
    pcm = sine_wave(converter.chunk_size, sample_rate=converter.input_rate).tobytes()
    first_result = converter.convert_chunk(pcm) or b""
    combined = first_result + (converter.flush() or b"")

    expected = run_conversion([pcm])
    assert expected
    assert combined == expected


def test_small_chunks_buffer_until_threshold(
    converter: AudioStreamConverter,
) -> None:
    tiny_chunk = sine_wave(
        max(1, converter.min_chunk_size // 2),
        sample_rate=converter.input_rate,
    ).tobytes()

    first_result = converter.convert_chunk(tiny_chunk)
    assert first_result is None

    second_chunk = sine_wave(
        converter.min_chunk_size, sample_rate=converter.input_rate
    ).tobytes()
    second_result = converter.convert_chunk(second_chunk)
    assert second_result is not None
    flushed = converter.flush() or b""

    expected = run_conversion([tiny_chunk, second_chunk])
    actual = (second_result or b"") + flushed
    assert actual == expected


def test_streaming_accumulates_and_flushes(
    converter: AudioStreamConverter,
) -> None:
    chunks_bytes = [
        sine_wave(converter.chunk_size, sample_rate=converter.input_rate).tobytes()
        for _ in range(5)
    ]

    outputs: list[bytes] = []
    for chunk in chunks_bytes:
        maybe = converter.convert_chunk(chunk)
        if maybe:
            outputs.append(maybe)
    flush = converter.flush()
    if flush:
        outputs.append(flush)

    emitted = b"".join(outputs)
    expected = run_conversion(chunks_bytes)
    assert emitted == expected
    assert emitted


@pytest.mark.parametrize("factors", [[1, 5, 2, 1], [3, 7], [2, 2, 2, 2, 2]])
def test_variable_chunk_sizes(
    converter: AudioStreamConverter,
    factors: list[int],
) -> None:
    base = converter.chunk_size // 2 or 1
    outputs: list[bytes] = []
    chunk_list: list[bytes] = []

    for factor in factors:
        samples = base * factor
        chunk = sine_wave(samples, sample_rate=converter.input_rate).tobytes()
        chunk_list.append(chunk)
        maybe_output = converter.convert_chunk(chunk)
        if maybe_output:
            outputs.append(maybe_output)

    final = converter.flush()
    if final:
        outputs.append(final)

    produced = b"".join(outputs)
    expected = run_conversion(chunk_list)
    assert produced == expected


def test_flush_returns_remaining_data(converter: AudioStreamConverter) -> None:
    tiny_chunk = sine_wave(
        max(1, converter.min_chunk_size // 4),
        sample_rate=converter.input_rate,
    ).tobytes()
    initial = converter.convert_chunk(tiny_chunk) or b""
    flushed = converter.flush()
    assert flushed is not None
    assert len(flushed) > 0 or len(initial) > 0
    assert converter.flush() is None

    expected = run_conversion([tiny_chunk])
    assert initial + flushed == expected


def test_convert_chunk_handles_odd_byte_length(
    converter: AudioStreamConverter,
) -> None:
    base_chunk = sine_wave(
        converter.chunk_size, sample_rate=converter.input_rate
    ).tobytes()
    odd_chunk = base_chunk + b"\x00"

    converter.convert_chunk(odd_chunk)
    buffered_bytes = converter._byte_buffer  # noqa: SLF001 (access for testing)
    assert len(buffered_bytes) == 1

    follow_up = sine_wave(
        converter.chunk_size, sample_rate=converter.input_rate
    ).tobytes()
    second = converter.convert_chunk(follow_up)
    assert second is not None


def test_reset_clears_internal_state(converter: AudioStreamConverter) -> None:
    converter.convert_chunk(
        sine_wave(converter.chunk_size, sample_rate=converter.input_rate).tobytes()
    )
    converter.reset()

    assert converter.residual_buffer.size == 0
    refreshed = converter.convert_chunk(
        sine_wave(converter.chunk_size, sample_rate=converter.input_rate).tobytes()
    )
    assert refreshed is not None


def test_silence_yields_constant_payload(converter: AudioStreamConverter) -> None:
    silence = np.zeros(converter.chunk_size, dtype=np.int16).tobytes()
    result = converter.convert_chunk(silence)

    assert result is not None
    assert len(set(result)) == 1


def test_convert_stream_accepts_iterables(converter: AudioStreamConverter) -> None:
    chunk_list = [
        sine_wave(converter.chunk_size, sample_rate=converter.input_rate).tobytes()
        for _ in range(3)
    ]
    outputs = list(converter.convert_stream(iter(chunk_list)))
    actual = b"".join(outputs) + (converter.flush() or b"")

    expected = run_conversion(chunk_list)
    assert actual == expected


def test_convert_file_roundtrip(tmp_path: Path) -> None:
    converter = AudioStreamConverter()
    samples = converter.chunk_size
    pcm = sine_wave(samples, sample_rate=converter.input_rate)

    input_path = tmp_path / "input.raw"
    output_path = tmp_path / "output.ulaw"
    input_path.write_bytes(pcm.tobytes())

    convert_file(input_path, output_path, converter=converter)

    # Reset converter for direct conversion comparison
    converter.reset()
    direct = converter.convert_chunk(pcm.tobytes()) or b""
    direct += converter.flush() or b""
    assert output_path.read_bytes() == direct


def test_ulaw_decoder_roundtrip() -> None:
    converter = AudioStreamConverter()
    decoder = MuLawStreamDecoder()

    pcm = sine_wave(converter.chunk_size * 3, sample_rate=converter.input_rate)
    ulaw = run_conversion([pcm.tobytes()])

    decoded_parts: list[bytes] = []
    chunk = decoder.chunk_size
    for offset in range(0, len(ulaw), chunk):
        part = decoder.convert_chunk(ulaw[offset : offset + chunk])
        if part:
            decoded_parts.append(part)
    final = decoder.flush()
    if final:
        decoded_parts.append(final)

    decoded = np.frombuffer(b"".join(decoded_parts), dtype=np.int16)
    original = pcm[: len(decoded)].astype(np.int32)
    assert decoded.size == original.size
    assert np.max(np.abs(decoded)) > 0

    reconverter = AudioStreamConverter()
    reencoded_parts: list[bytes] = []
    step = reconverter.chunk_size * 2
    decoded_bytes = decoded.tobytes()
    for offset in range(0, len(decoded_bytes), step):
        part = reconverter.convert_chunk(decoded_bytes[offset : offset + step])
        if part:
            reencoded_parts.append(part)
    final_ulaw = reconverter.flush()
    if final_ulaw:
        reencoded_parts.append(final_ulaw)

    reencoded = b"".join(reencoded_parts)
    assert reencoded
    tolerance = max(reconverter.down_factor, 16)
    assert abs(len(reencoded) - len(ulaw)) <= tolerance


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_rate": 0},
        {"output_rate": 0},
        {"chunk_size": 0},
        {"input_rate": -1},
        {"chunk_size": -100},
    ],
)
def test_invalid_converter_parameters_raise_value_error(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        AudioStreamConverter(**kwargs)


def test_convert_file_validates_chunk_size(tmp_path: Path) -> None:
    source = tmp_path / "raw.pcm"
    target = tmp_path / "output.ulaw"
    source.write_bytes(b"\x00\x00" * 10)

    with pytest.raises(ValueError):
        convert_file(source, target, chunk_samples=0)


def test_convert_ulaw_file_roundtrip(tmp_path: Path) -> None:
    converter = AudioStreamConverter()
    decoder = MuLawStreamDecoder()

    pcm_samples = sine_wave(converter.chunk_size * 2, sample_rate=converter.input_rate)
    pcm_path = tmp_path / "input.raw"
    ulaw_path = tmp_path / "encoded.ulaw"
    decoded_path = tmp_path / "decoded.raw"
    pcm_path.write_bytes(pcm_samples.tobytes())

    convert_file(pcm_path, ulaw_path, converter=converter)
    convert_ulaw_file(ulaw_path, decoded_path, decoder=decoder)

    decoded_bytes = decoded_path.read_bytes()
    decoded = np.frombuffer(decoded_bytes, dtype=np.int16)
    assert decoded.size > 0

    reconverter = AudioStreamConverter()
    re_ulaw_parts: list[bytes] = []
    step = reconverter.chunk_size * 2
    for offset in range(0, len(decoded_bytes), step):
        part = reconverter.convert_chunk(decoded_bytes[offset : offset + step])
        if part:
            re_ulaw_parts.append(part)
    final_part = reconverter.flush()
    if final_part:
        re_ulaw_parts.append(final_part)

    re_ulaw = b"".join(re_ulaw_parts)
    assert re_ulaw
    tolerance = max(reconverter.down_factor, 16)
    assert abs(len(re_ulaw) - len(ulaw_path.read_bytes())) <= tolerance


def test_sample_wav_conversion(tmp_path: Path) -> None:
    pcm16 = load_sample_pcm16(limit_seconds=1.0)
    converter = AudioStreamConverter()

    ulaw_chunks: list[bytes] = []
    for start in range(0, len(pcm16), converter.chunk_size):
        end = start + converter.chunk_size
        chunk = converter.convert_chunk(pcm16[start:end].tobytes())
        if chunk:
            ulaw_chunks.append(chunk)

    final = converter.flush()
    if final:
        ulaw_chunks.append(final)

    ulaw_bytes = b"".join(ulaw_chunks)
    assert ulaw_bytes

    output_path = tmp_path / "sample.ulaw"
    output_path.write_bytes(ulaw_bytes)
    assert output_path.stat().st_size == len(ulaw_bytes)

    decoder = MuLawStreamDecoder()
    pcm_parts: list[bytes] = []
    chunk_size = decoder.chunk_size
    for start in range(0, len(ulaw_bytes), chunk_size):
        chunk = decoder.convert_chunk(ulaw_bytes[start : start + chunk_size])
        if chunk:
            pcm_parts.append(chunk)

    final = decoder.flush()
    if final:
        pcm_parts.append(final)

    pcm16_bytes = b"".join(pcm_parts)
    decoded_path = tmp_path / "sample_decoded.wav"
    with wave.open(str(decoded_path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(decoder.output_rate)
        wav_out.writeframes(pcm16_bytes)

    with wave.open(str(decoded_path), "rb") as wav_in:
        assert wav_in.getnchannels() == 1
        assert wav_in.getframerate() == decoder.output_rate
        expected_frames = len(ulaw_bytes) * decoder.output_rate / decoder.input_rate
        tolerance = max(decoder.down_factor, 16)
        assert abs(wav_in.getnframes() - expected_frames) <= tolerance
        frames = wav_in.readframes(wav_in.getnframes())
        pcm_samples = np.frombuffer(frames, dtype=np.int16)
        assert np.max(np.abs(pcm_samples)) > 0
