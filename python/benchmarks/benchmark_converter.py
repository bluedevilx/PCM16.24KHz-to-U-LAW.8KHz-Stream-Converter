"""Benchmark harness for AudioStreamConverter.

Usage:
    python benchmark_converter.py --duration 5 --chunk-ms 20 --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Iterable

import numpy as np

from audio_converter import AudioStreamConverter


def generate_stream(
    duration_sec: float,
    chunk_ms: float,
    sample_rate: int,
) -> list[bytes]:
    """Generate random PCM16 audio chunks for benchmarking."""
    total_samples = int(duration_sec * sample_rate)
    chunk_samples = max(1, int(sample_rate * (chunk_ms / 1000.0)))

    samples = np.random.randint(-32768, 32767, total_samples, dtype=np.int16)
    chunks: list[bytes] = []

    for start in range(0, total_samples, chunk_samples):
        chunk = samples[start : start + chunk_samples]
        if chunk.size == 0:
            continue
        chunks.append(chunk.tobytes())

    return chunks


def run_once(chunks: Iterable[bytes]) -> float:
    """Run converter over provided chunks once and return elapsed seconds."""
    converter = AudioStreamConverter()
    start = time.perf_counter()

    for chunk in chunks:
        converter.convert_chunk(chunk)

    converter.flush()
    return time.perf_counter() - start


def format_results(elapsed_sec: float, timings: list[float]) -> str:
    """Return formatted benchmark summary."""
    real_time_factors = [elapsed_sec / t for t in timings]
    avg_rtf = statistics.mean(real_time_factors)
    min_rtf = min(real_time_factors)
    max_rtf = max(real_time_factors)

    avg_time = statistics.mean(timings)
    stdev_time = statistics.pstdev(timings) if len(timings) > 1 else 0.0

    return (
        f"Runs: {len(timings)}\n"
        f"Avg wall time: {avg_time * 1000:.2f} ms "
        f"(σ={stdev_time * 1000:.2f} ms)\n"
        f"Real-time factor: avg={avg_rtf:.2f}×, "
        f"min={min_rtf:.2f}×, max={max_rtf:.2f}×\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the audio converter throughput."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of synthetic audio to process in seconds (default: 10s).",
    )
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=20.0,
        help="Chunk size in milliseconds (default: 20ms).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repetitions to average (default: 5).",
    )

    args = parser.parse_args()

    converter = AudioStreamConverter()
    total_audio_seconds = args.duration
    chunks = generate_stream(total_audio_seconds, args.chunk_ms, converter.input_rate)

    # Warm-up run
    run_once(chunks)

    timings = [run_once(chunks) for _ in range(max(1, args.repeats))]

    print(
        "Configuration: "
        f"duration={total_audio_seconds}s, "
        f"chunk={args.chunk_ms}ms, "
        f"repeats={len(timings)}"
    )
    print(format_results(total_audio_seconds, timings))


if __name__ == "__main__":
    main()
