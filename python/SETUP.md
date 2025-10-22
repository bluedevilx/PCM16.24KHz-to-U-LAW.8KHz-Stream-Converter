# Setup Guide

Quick setup guide for the Audio Stream Converter plugin.

## Prerequisites

- Python 3.8 or higher (tested with Python 3.13)
- pip (Python package manager)

## Installation

> From the repository root, run `cd python` before executing the steps below.

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Package (editable for development)

```bash
pip install -e .[dev]
```

### 3. Verify Installation

```bash
pytest
ruff check .
ruff format --check .
mypy src
```

You should see all quality gates passing. Example output:
```
============================= test session starts ==============================
collected 18 items

tests/test_converter.py ..................

============================== 18 passed in XX.XXs =============================
```

## Quick Test

### Run Examples

```bash
# Run basic examples
python examples/example_gemini_twilio.py

# Run integration examples
python examples/integration_example.py

# Convert bundled sample WAV to μ-law output
python examples/convert_sample.py
# Uses the first channel of the multichannel capture, resamples, and exports:
#   -> examples/data/down_sample_8khz.ulaw
#   -> examples/data/down_sample_8khz.wav (8KHz PCM)
#   -> examples/data/up_sample_pcm16_24khz_decoded.wav (24KHz PCM)

# Reverse conversion (μ-law 8KHz → PCM16 24KHz)
audio-stream-converter --direction ulaw-to-pcm --input-rate 8000 --output-rate 24000 \
    examples/data/down_sample_8khz.ulaw examples/data/sample_pcm16_24khz.raw

## Node.js Library

```bash
cd node
npm install
npm run build
npm run sample  # mirrors the Python convert_sample script
```
```

### Test with Sample Audio

Create a simple test:

```python
from audio_converter import AudioStreamConverter
import numpy as np

# Create converter
converter = AudioStreamConverter()

# Generate 200ms of test audio (1000 Hz tone)
t = np.linspace(0, 0.2, 4800, endpoint=False)  # 4800 samples at 24KHz
sine = (16000 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)

# Convert
ulaw_data = converter.convert_chunk(sine.tobytes())

print(f"Input: {len(sine.tobytes())} bytes PCM16")
print(f"Output: {len(ulaw_data)} bytes μ-law")
```

## Integration with Gemini Live API

### Install Gemini SDK

```bash
pip install google-generativeai
```

### Basic Integration

```python
import google.generativeai as genai
from audio_converter import AudioStreamConverter

# Configure Gemini
genai.configure(api_key='YOUR_API_KEY')
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Create converter
converter = AudioStreamConverter()

# Use in your streaming loop
async for response in model.stream_generate_content(...):
    if hasattr(response, 'audio'):
        pcm_data = response.audio.data
        ulaw_data = converter.convert_chunk(pcm_data)
        if ulaw_data:
            # Send to Twilio or other service
            pass
```

## Integration with Twilio

### Install Twilio SDK

```bash
pip install twilio fastapi uvicorn websockets
```

### Basic Integration

```python
from fastapi import FastAPI, WebSocket
from audio_converter import AudioStreamConverter
import base64
import json

app = FastAPI()

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    converter = AudioStreamConverter()

    async for message in websocket.iter_text():
        data = json.loads(message)

        if data['event'] == 'media':
            # Receive from Gemini (PCM16 24KHz)
            pcm_audio = get_from_gemini()

            # Convert
            ulaw_audio = converter.convert_chunk(pcm_audio)

            if ulaw_audio:
                # Send to Twilio
                payload = base64.b64encode(ulaw_audio).decode('utf-8')
                await websocket.send_json({
                    "event": "media",
                    "media": {"payload": payload}
                })

# Run: uvicorn your_script:app --reload
```

## Troubleshooting

### Import Error: No module named 'numpy'

**Solution**: Make sure you activated the virtual environment and installed dependencies:
```bash
source venv/bin/activate
pip install -e .[dev]
```

### Test Failures

**Solution**: Ensure you're using Python 3.8+:
```bash
python --version
```

### Performance Issues

**Solution**: Adjust chunk size for your latency requirements:
```python
# Lower latency (smaller chunks)
converter = AudioStreamConverter(chunk_size=2400)  # 100ms

# Standard (balanced)
converter = AudioStreamConverter(chunk_size=4800)  # 200ms

# Lower CPU (larger chunks)
converter = AudioStreamConverter(chunk_size=9600)  # 400ms
```

## Next Steps

1. Read the [README.md](README.md) for detailed API documentation
2. Review [example_gemini_twilio.py](example_gemini_twilio.py) for usage patterns
3. Check [integration_example.py](integration_example.py) for production examples
4. Implement your specific use case!

## Support

For issues or questions:
- Review the examples in this repository
- Check the API documentation in README.md
- Verify your audio format matches the expected input (PCM16 24KHz)

## License

MIT License - use freely in your projects!
