"""
Example: Streaming audio conversion from Gemini Live API to Twilio

This example demonstrates how to use the AudioStreamConverter to bridge
audio between Gemini Live API (PCM16 24KHz) and Twilio (μ-law 8KHz).
"""

import asyncio

from audio_converter import AudioStreamConverter, MuLawStreamDecoder


# Example 1: Simple synchronous conversion
def example_simple_conversion():
    """
    Simple example: convert chunks as they arrive
    """
    converter = AudioStreamConverter()

    # Simulate receiving chunks from Gemini Live API
    gemini_chunks = [
        b"\x00\x01" * 4800,  # Mock PCM16 data (4800 samples = 200ms at 24KHz)
        b"\x00\x02" * 4800,
        b"\x00\x03" * 4800,
    ]

    print("Converting audio chunks...")
    for chunk_idx, chunk in enumerate(gemini_chunks):
        ulaw_chunk = converter.convert_chunk(chunk)
        if ulaw_chunk:
            print(
                f"Chunk {chunk_idx}: {len(chunk)} bytes PCM16 "
                f"→ {len(ulaw_chunk)} bytes μ-law"
            )
            # Send to Twilio here
            # twilio_stream.send(ulaw_chunk)

    # Don't forget to flush at the end
    final_chunk = converter.flush()
    if final_chunk:
        print(f"Final: {len(final_chunk)} bytes μ-law")


# Example 2: Generator-based streaming
def example_generator_stream():
    """
    Example using generator for continuous streaming
    """
    converter = AudioStreamConverter()

    def gemini_audio_stream():
        """Mock generator simulating Gemini Live API stream"""
        for _ in range(10):
            # Simulate receiving audio chunks
            yield b"\x00\x01" * 4800

    print("\nStreaming conversion with generator...")
    chunk_count = 0
    for ulaw_chunk in converter.convert_stream(gemini_audio_stream()):
        chunk_count += 1
        print(f"Converted chunk {chunk_count}: {len(ulaw_chunk)} bytes")
        # Send to Twilio
        # await twilio_client.send(ulaw_chunk)


# Example 3: Async streaming (typical for WebSocket connections)
async def example_async_websocket():
    """
    Example with async/await for WebSocket-based streaming
    (typical pattern for both Gemini and Twilio)
    """
    converter = AudioStreamConverter()

    # Mock WebSocket clients
    class MockWebSocket:
        async def receive_bytes(self):
            # Simulate receiving from Gemini
            await asyncio.sleep(0.2)  # Simulate network delay
            return b"\x00\x01" * 4800

        async def send_bytes(self, data):
            # Simulate sending to Twilio
            print(f"  → Sent {len(data)} bytes to Twilio")

    gemini_ws = MockWebSocket()
    twilio_ws = MockWebSocket()

    print("\nAsync WebSocket streaming...")
    try:
        for i in range(5):
            # Receive from Gemini Live API
            pcm_chunk = await gemini_ws.receive_bytes()
            print(f"Received chunk {i} from Gemini: {len(pcm_chunk)} bytes")

            # Convert
            ulaw_chunk = converter.convert_chunk(pcm_chunk)

            if ulaw_chunk:
                # Send to Twilio
                await twilio_ws.send_bytes(ulaw_chunk)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Flush any remaining audio
        final_chunk = converter.flush()
        if final_chunk:
            await twilio_ws.send_bytes(final_chunk)
            print("Flushed final chunk")


# Example 4: Real Twilio integration structure
async def example_twilio_integration():
    """
    Example structure for real Twilio Media Stream integration
    """
    print("\nTwilio Media Stream integration example...")
    print("This shows the structure for a real implementation:\n")

    # Pseudo-code for Twilio Media Stream
    """
    from twilio.twiml.voice_response import VoiceResponse, Start
    from fastapi import WebSocket

    @app.websocket("/media-stream")
    async def media_stream(websocket: WebSocket):
        await websocket.accept()
        converter = AudioStreamConverter()

        # Receive from Gemini, convert, send to Twilio
        async def gemini_to_twilio():
            async for gemini_audio in gemini_websocket:
                # gemini_audio is PCM16 24KHz from Gemini Live API
                ulaw_audio = converter.convert_chunk(gemini_audio)

                if ulaw_audio:
                    # Create Twilio media message
                    message = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {
                            "payload": base64.b64encode(ulaw_audio).decode()
                        }
                    }
                    await websocket.send_json(message)

        # Start bidirectional streaming
        await gemini_to_twilio()
    """

    print("Key points for Twilio integration:")
    print("1. Twilio expects μ-law 8KHz audio")
    print("2. Send as base64-encoded in 'media' events")
    print("3. Use WebSocket for bidirectional streaming")
    print("4. StreamSid identifies the call")


# Example 5: Handling variable chunk sizes
def example_variable_chunks():
    """
    Example handling variable-sized chunks from network
    """
    converter = AudioStreamConverter()

    # Simulate variable-sized chunks (realistic for network streaming)
    variable_chunks = [
        b"\x00\x01" * 1000,  # Small chunk
        b"\x00\x02" * 8000,  # Large chunk
        b"\x00\x03" * 500,  # Very small chunk
        b"\x00\x04" * 4800,  # Normal chunk
    ]

    print("\nHandling variable-sized chunks...")
    for i, chunk in enumerate(variable_chunks):
        ulaw_chunk = converter.convert_chunk(chunk)
        if ulaw_chunk:
            input_samples = len(chunk) // 2
            output_samples = len(ulaw_chunk)
            duration_ms = (input_samples / 24000) * 1000
            print(
                f"Chunk {i}: {input_samples} samples "
                f"({duration_ms:.1f}ms) → {output_samples} μ-law bytes"
            )
        else:
            print(f"Chunk {i}: Buffered (too small)")

    # Flush remaining
    final = converter.flush()
    if final:
        print(f"Flushed: {len(final)} bytes")


def example_ulaw_to_pcm():
    """Demonstrate decoding μ-law 8KHz audio back to PCM16 24KHz."""
    decoder = MuLawStreamDecoder()

    # Mock μ-law chunks (would normally come from Twilio)
    ulaw_chunks = [b"\xff\xff" * decoder.chunk_size for _ in range(3)]

    print("\nDecoding μ-law chunks back to PCM16...")
    total_pcm = 0
    for idx, chunk in enumerate(ulaw_chunks):
        pcm = decoder.convert_chunk(chunk)
        if pcm:
            total_pcm += len(pcm)
            print(
                f"Decoded chunk {idx}: {len(chunk)} bytes μ-law → "
                f"{len(pcm)} bytes PCM"
            )

    final_pcm = decoder.flush()
    if final_pcm:
        total_pcm += len(final_pcm)
        print(f"Flushed: {len(final_pcm)} bytes PCM")

    print(f"Total decoded PCM bytes: {total_pcm}")


if __name__ == "__main__":
    # Run examples
    example_simple_conversion()
    example_generator_stream()
    example_variable_chunks()
    example_ulaw_to_pcm()

    # Run async example
    print("\nRunning async example...")
    asyncio.run(example_async_websocket())

    # Show integration structure
    asyncio.run(example_twilio_integration())

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
