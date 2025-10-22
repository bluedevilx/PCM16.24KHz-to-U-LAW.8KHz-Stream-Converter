# Audio Stream Converter: PCM16 24KHz → μ-law 8KHz

A Python plugin for converting streaming audio from Gemini Live API format (PCM16 24KHz) to Twilio-compatible format (μ-law 8KHz) in real-time.

## Features

- **Real-time streaming conversion**: Process audio chunks as they arrive
- **High-quality downsampling**: Uses scipy's polyphase resampling with anti-aliasing
- **Standard μ-law encoding**: Compatible with telephony systems (G.711)
- **Stateful processing**: Handles chunk boundaries smoothly with internal buffering
- **Flexible interface**: Supports both synchronous and asynchronous streaming patterns
- **Bidirectional conversion**: Stream μ-law 8KHz audio back to PCM16 24KHz using the built-in decoder
- **Production-grade tooling**: Fully type annotated, CLI-enabled, and backed by an automated test suite

## Installation

> All commands in this document assume you are inside the `python/` directory (e.g. `cd python`).

```bash
pip install audio-stream-converter
```

For local development:

```bash
pip install -e .[dev]
```

## Quick Start

### Basic Usage

```python
from audio_converter import AudioStreamConverter

# Initialize converter
converter = AudioStreamConverter()

# Convert a single chunk
pcm16_chunk = receive_from_gemini()  # Your PCM16 24KHz data
ulaw_chunk = converter.convert_chunk(pcm16_chunk)

if ulaw_chunk:
    send_to_twilio(ulaw_chunk)  # Send μ-law 8KHz to Twilio

# At end of stream
final_chunk = converter.flush()
if final_chunk:
    send_to_twilio(final_chunk)
```

### Command Line Conversion

Convert entire PCM files without writing code:

```bash
audio-stream-converter input_24khz.raw output_8khz.ulaw

# Reverse conversion: μ-law 8KHz to PCM16 24KHz
audio-stream-converter --direction ulaw-to-pcm --input-rate 8000 --output-rate 24000 \
    input_8khz.ulaw output_24khz.raw
```

### Generator-based Streaming

```python
# Stream conversion using generators
for ulaw_chunk in converter.convert_stream(gemini_audio_stream):
    send_to_twilio(ulaw_chunk)
```

### Async WebSocket Example

```python
async def bridge_audio(gemini_ws, twilio_ws):
    converter = AudioStreamConverter()

    async for pcm_data in gemini_ws:
        ulaw_data = converter.convert_chunk(pcm_data)
        if ulaw_data:
            await twilio_ws.send(ulaw_data)

    # Flush remaining audio
    final = converter.flush()
    if final:
        await twilio_ws.send(final)
```

### Whole File Conversion

```python
from pathlib import Path
from audio_converter import convert_file

convert_file(
    input_path=Path("input_24khz.raw"),
    output_path=Path("output_8khz.ulaw"),
    chunk_samples=4800,  # optional override
)
```

### Reverse Conversion (μ-law → PCM16)

```python
from audio_converter import MuLawStreamDecoder

decoder = MuLawStreamDecoder()

# μ-law chunk from Twilio
ulaw_chunk = receive_from_twilio()
pcm_chunk = decoder.convert_chunk(ulaw_chunk)

if pcm_chunk:
    send_to_gemini(pcm_chunk)  # PCM16 24KHz

final_pcm = decoder.flush()
if final_pcm:
    send_to_gemini(final_pcm)
```

## Command Line Interface

An `audio-stream-converter` console script ships with the package:

```bash
$ audio-stream-converter --help
usage: audio-stream-converter [-h] [--chunk-samples CHUNK_SAMPLES]
                              [--input-rate INPUT_RATE] [--output-rate OUTPUT_RATE]
                              [--log-level {critical,debug,error,info,warning}] [--version]
                              input_path output_path
```

- `--chunk-samples`: Number of PCM samples per chunk (defaults to 4800).
- `--direction`: Choose `pcm-to-ulaw` or `ulaw-to-pcm` (default: `pcm-to-ulaw`).
- `--input-rate` / `--output-rate`: Override sample rates when upstream services change.
- `--log-level`: Choose from `critical`, `error`, `warning`, `info`, or `debug`.

### Bundled Sample Conversion

A multi-channel WAV recording lives at `examples/data/sample_pcm16_24khz.wav`. Convert
it to μ-law using the helper script:

```bash
python examples/convert_sample.py
# Produces examples/data/down_sample_8khz.ulaw
# and decoded WAVs:
#   - examples/data/down_sample_8khz.wav
#   - examples/data/up_sample_pcm16_24khz_decoded.wav
```

### Node.js Package

A TypeScript implementation lives under [`node/`](node/). Build and run the sample conversion:

```bash
cd node
npm install
npm run sample
```

This produces the same μ-law and decoded WAV files as the Python tooling.

The script normalises, selects the first channel (the cleanest voice capture), resamples
to 24KHz PCM16, then streams the result through `AudioStreamConverter`. It also decodes
the μ-law bytes back to an
8KHz PCM WAV so you can preview the telephony-quality output locally.

## How It Works

### Audio Conversion Pipeline

```
Gemini Live API          Converter                Twilio
─────────────────        ────────────             ──────
PCM16, 24KHz      →      Downsample        →      μ-law, 8KHz
(16-bit linear)          (3:1 ratio)              (8-bit compressed)
2 bytes/sample           Anti-aliasing            1 byte/sample
```

### Processing Steps

1. **Downsampling (24KHz → 8KHz)**
   - Uses scipy's `resample_poly` with polyphase filtering
   - Automatically applies anti-aliasing filter
   - Reduces sample rate by factor of 3

2. **Format Conversion (PCM16 → μ-law)**
   - Converts 16-bit linear PCM to 8-bit μ-law
   - Uses standard G.711 μ-law algorithm
   - Reduces bandwidth by 50%

3. **Buffer Management**
   - Maintains internal buffer for smooth chunk transitions
   - Handles variable-sized input chunks
   - Prevents audio artifacts at chunk boundaries

## Repository Layout

- `src/audio_converter/` – core library package (`AudioStreamConverter`, CLI entry point)
- `benchmarks/` – synthetic workload benchmarks (`benchmark_converter.py`)
- `examples/` – integration samples for Gemini/Twilio bridges
- `examples/data/` – bundled sample audio (`sample_pcm16_24khz.wav`, `down_sample_8khz.ulaw`, etc.)
- `tests/` – regression test suite (`tests/test_converter.py`)

## Testing

```bash
pytest
ruff check .
ruff format --check .
mypy src
```

## API Reference

### AudioStreamConverter

Main class for audio conversion.

#### Constructor

```python
AudioStreamConverter(
    input_rate: int = 24000,
    output_rate: int = 8000,
    chunk_size: int = 4800
)
```

**Parameters:**
- `input_rate`: Input sample rate in Hz (default: 24000)
- `output_rate`: Output sample rate in Hz (default: 8000)
- `chunk_size`: Expected chunk size in samples (default: 4800 = 200ms)

#### Methods

##### `convert_chunk(pcm16_data: bytes) -> Optional[bytes]`

Convert a single chunk of PCM16 audio to μ-law.

**Parameters:**
- `pcm16_data`: Raw PCM16 audio bytes (little-endian)

**Returns:**
- μ-law encoded bytes, or `None` if insufficient data

##### `convert_stream(audio_stream: Iterator[bytes]) -> Iterator[bytes]`

Convert a stream of audio chunks (generator).

**Parameters:**
- `audio_stream`: Iterator yielding PCM16 chunks

**Yields:**
- μ-law encoded chunks

##### `flush() -> Optional[bytes]`

Process any remaining buffered audio.

**Returns:**
- Final μ-law chunk, or `None` if buffer empty

##### `reset()`

Clear all internal buffers. Call before processing a new stream.

##### `convert_file(input_path, output_path, *, chunk_samples=None, converter=None)`

Convert entire PCM16 files via the streaming pipeline. If you supply a custom converter it will be reset before use. `chunk_samples` controls how many PCM samples to read per iteration (defaults to the converter chunk size).

### MuLawStreamDecoder

Inverse companion to `AudioStreamConverter`, decoding μ-law 8KHz audio to PCM16 24KHz.

```python
from audio_converter import MuLawStreamDecoder

decoder = MuLawStreamDecoder(
    input_rate=8000,
    output_rate=24000,
    chunk_size=1600,
)
```

#### Methods

- `convert_chunk(ulaw_data: bytes) -> Optional[bytes]`
- `convert_stream(audio_stream: Iterable[bytes]) -> Iterator[bytes]`
- `flush() -> Optional[bytes]`
- `reset()`

#### Helper

- `convert_ulaw_file(input_path, output_path, *, chunk_samples=None, decoder=None)`
  converts μ-law files back to PCM16 using the streaming decoder.

## Integration Examples

### Twilio Media Streams

```python
import base64
from audio_converter import AudioStreamConverter, MuLawStreamDecoder
from fastapi import WebSocket

@app.websocket("/twilio-media")
async def handle_twilio_media(websocket: WebSocket):
    await websocket.accept()
    converter = AudioStreamConverter()
    decoder = MuLawStreamDecoder()

    async for message in websocket.iter_json():
        event = message.get("event")
        if event == "media":
            # μ-law → PCM16 for Gemini
            pcm = decoder.convert_chunk(base64.b64decode(message["media"]["payload"]))
            if pcm:
                await forward_to_gemini(pcm)
        elif event == "gemini_audio":
            ulaw_audio = converter.convert_chunk(message["audio"])
            if ulaw_audio:
                await websocket.send_json({
                    "event": "media",
                    "streamSid": message["streamSid"],
                    "media": {"payload": base64.b64encode(ulaw_audio).decode("utf-8")}
                })
```

### Gemini Live API

```python
import google.generativeai as genai

async def stream_from_gemini():
    converter = AudioStreamConverter()

    # Connect to Gemini Live API
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    async for response in model.stream_generate_content(...):
        if hasattr(response, 'audio'):
            # Gemini audio is PCM16 24KHz
            pcm_data = response.audio.data

            # Convert to μ-law 8KHz
            ulaw_data = converter.convert_chunk(pcm_data)

            if ulaw_data:
                yield ulaw_data
```

## Performance Considerations

### Latency

- **Processing time**: ~1-2ms per 200ms audio chunk (on modern CPU)
- **Chunk size**: Smaller chunks = lower latency but more overhead
- **Recommended**: 100-200ms chunks (2400-4800 samples at 24KHz)

### Memory Usage

- **Buffer overhead**: Minimal (~10KB per converter instance)
- **Per-chunk memory**: Input size + output size (~10KB for 200ms)

### CPU Usage

- **Resampling**: Most CPU-intensive operation
- **Optimization**: scipy uses optimized C/Fortran libraries
- **Concurrent streams**: Each converter is independent, can run in parallel

### Benchmarking

Use the included harness to profile throughput:

```bash
python benchmarks/benchmark_converter.py --duration 5 --chunk-ms 20 --repeats 3
```

## Audio Quality

- **Sample rate reduction**: 24KHz → 8KHz (standard for telephony)
- **Frequency response**: 0-4KHz preserved (Nyquist limit at 8KHz)
- **Dynamic range**: ~14-bit effective (μ-law encoding)
- **THD+N**: <1% (typical for μ-law)

## Troubleshooting

### "No output from convert_chunk"

**Cause**: Input chunks too small, being buffered

**Solution**: Send larger chunks or call `flush()` at end

### "Audio sounds distorted"

**Cause**: Possible sample rate mismatch

**Solution**: Verify input is actually 24KHz PCM16

### "Latency too high"

**Cause**: Large chunk sizes

**Solution**: Reduce chunk_size parameter (e.g., 2400 = 100ms)

## Testing

Run the examples and tests:

```bash
# Run regression tests
pytest

# Quality gates
ruff check .
ruff format --check .
mypy src

# Example integrations
python examples/example_gemini_twilio.py
python examples/integration_example.py

# File conversion via CLI
audio-stream-converter input_24khz.raw output_8khz.ulaw
```

## Technical Details

### Resampling Algorithm

Uses scipy's `resample_poly` which implements:
- Polyphase filtering for efficiency
- Kaiser window for anti-aliasing
- Integer upsampling/downsampling ratio (24000/8000 = 3)

### μ-law Encoding

Standard ITU-T G.711 μ-law:
- Logarithmic companding
- 8-bit output (256 levels)
- ~14-bit dynamic range
- Optimized for speech

## License

MIT License - feel free to use in your projects

## Development Workflow

1. Create a virtual environment and install dev dependencies: `pip install -e .[dev]`
2. Run `ruff check .` and `ruff format --check .` to enforce style and imports.
3. Run `mypy src` to validate type hints (the package ships `py.typed`).
4. Run `pytest --cov` to execute the regression suite and collect coverage.

## Contributing

Contributions welcome! Please ensure:
- Code passes `ruff`, `mypy`, and `pytest`
- New behaviour ships with accompanying tests
- Documentation stays current with API changes

## Support

For issues or questions:
1. Check the examples in `example_gemini_twilio.py`
2. Review this README
3. Open an issue on GitHub
