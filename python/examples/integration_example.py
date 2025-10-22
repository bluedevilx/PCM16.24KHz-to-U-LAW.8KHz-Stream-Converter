"""
Practical Integration Example: Gemini Live API + Twilio

This example shows how to integrate the AudioStreamConverter with
Gemini Live API and Twilio in a production-like setup.

Requirements:
    pip install google-generativeai twilio fastapi uvicorn websockets
"""

import asyncio
import base64  # noqa: F401  (referenced in documentation snippets)
import json  # noqa: F401 (referenced in documentation snippets)

from audio_converter import AudioStreamConverter, MuLawStreamDecoder


# Example 1: Gemini Live API to Twilio Bridge
async def gemini_to_twilio_bridge():
    """
    Bridge audio from Gemini Live API to Twilio Media Streams

    This is a simplified example showing the core logic.
    In production, you'd handle authentication, error recovery, etc.
    """

    # Initialize converter
    _converter = AudioStreamConverter()

    # Pseudo-code structure (replace with actual API calls)
    print("Gemini → Twilio Bridge Example")
    print("-" * 40)

    """
    # 1. Connect to Gemini Live API
    import google.generativeai as genai

    genai.configure(api_key='YOUR_API_KEY')
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    # Start live session
    live_session = model.start_chat(enable_audio=True)

    # 2. Connect to Twilio WebSocket
    import websockets

    twilio_ws = await websockets.connect('wss://your-twilio-endpoint')

    # 3. Stream audio from Gemini to Twilio
    async for response in live_session.stream_content():
        if hasattr(response, 'audio'):
            # Gemini provides PCM16 at 24KHz
            pcm_audio = response.audio.data

            # Convert to μ-law 8KHz
            ulaw_audio = converter.convert_chunk(pcm_audio)

            if ulaw_audio:
                # Encode for Twilio Media Stream
                payload = base64.b64encode(ulaw_audio).decode('utf-8')

                # Send to Twilio
                message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": payload
                    }
                }

                await twilio_ws.send(json.dumps(message))

    # Flush any remaining audio
    final_audio = converter.flush()
    if final_audio:
        payload = base64.b64encode(final_audio).decode('utf-8')
        message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload}
        }
        await twilio_ws.send(json.dumps(message))
    """

    print("See code comments for implementation details")
    print()


# Example 2: FastAPI WebSocket Server for Twilio
def create_twilio_websocket_server():
    """
    FastAPI server that receives Twilio Media Streams and
    forwards to Gemini after processing
    """

    print("FastAPI Twilio Server Example")
    print("-" * 40)

    """
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import Response

    app = FastAPI()

    # Dictionary to store active converters by call SID
    active_converters = {}

    @app.post("/voice")
    async def handle_incoming_call():
        '''Handle incoming Twilio voice call'''
        from twilio.twiml.voice_response import VoiceResponse, Start

        response = VoiceResponse()

        # Start media stream
        start = Start()
        start.stream(url='wss://your-domain/media-stream')
        response.append(start)

        # Pause to keep call active
        response.pause(length=60)

        return Response(content=str(response), media_type='application/xml')

    @app.websocket("/media-stream")
    async def media_stream_handler(websocket: WebSocket):
        '''Handle Twilio Media Stream WebSocket'''
        await websocket.accept()

        converter = None
        stream_sid = None

        try:
            async for message in websocket.iter_text():
                data = json.loads(message)
                event = data.get('event')

                if event == 'start':
                    # New stream started
                    stream_sid = data['start']['streamSid']
                    converter = AudioStreamConverter()
                    active_converters[stream_sid] = converter
                    print(f"Stream started: {stream_sid}")

                elif event == 'media':
                    # Received audio from Twilio (μ-law 8KHz from caller)
                    payload = data['media']['payload']
                    audio_data = base64.b64decode(payload)

                    # Here you would:
                    # 1. Convert Twilio audio (μ-law 8KHz) to format for Gemini
                    # 2. Send to Gemini Live API
                    # 3. Receive response from Gemini
                    # 4. Use our converter to send back to Twilio
                    # (see bidirectional example below)

                elif event == 'stop':
                    # Stream ended
                    if converter:
                        final = converter.flush()
                        # Send final audio if needed

                    if stream_sid in active_converters:
                        del active_converters[stream_sid]

                    print(f"Stream stopped: {stream_sid}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            await websocket.close()

    # Run with: uvicorn integration_example:app --reload
    """

    print("See code comments for implementation details")
    print()


# Example 3: Bidirectional Audio Bridge
async def bidirectional_audio_bridge():
    """
    Bidirectional bridge: Twilio caller ↔ Gemini Live API

    This handles both directions:
    - Caller audio (Twilio) → Gemini (needs upsampling)
    - Gemini audio → Twilio (our converter)
    """

    print("Bidirectional Bridge Example")
    print("-" * 40)

    # Converter for Gemini → Twilio
    _gemini_to_twilio = AudioStreamConverter(input_rate=24000, output_rate=8000)

    # For Twilio → Gemini, use the decoder to upsample μ-law audio
    _twilio_to_gemini = MuLawStreamDecoder(input_rate=8000, output_rate=24000)

    """
    async def handle_bidirectional_stream(twilio_ws, gemini_session):
        # Task 1: Twilio → Gemini
        twilio_to_gemini = MuLawStreamDecoder(input_rate=8000, output_rate=24000)
        async def twilio_to_gemini():
            async for message in twilio_ws:
                data = json.loads(message)

                if data['event'] == 'media':
                    # Get μ-law audio from Twilio
                    ulaw_data = base64.b64decode(data['media']['payload'])

                    # Convert μ-law 8KHz → PCM16 24KHz
                    pcm_data = twilio_to_gemini.convert_chunk(ulaw_data)

                    if pcm_data:
                        await gemini_session.send_audio(pcm_data)

            final_pcm = twilio_to_gemini.flush()
            if final_pcm:
                await gemini_session.send_audio(final_pcm)

        # Task 2: Gemini → Twilio
        async def gemini_to_twilio_stream():
            async for response in gemini_session.stream_content():
                if hasattr(response, 'audio'):
                    # Get PCM16 24KHz from Gemini
                    pcm_data = response.audio.data

                    # Convert PCM16 24KHz → μ-law 8KHz (our converter!)
                    ulaw_data = gemini_to_twilio.convert_chunk(pcm_data)

                    if ulaw_data:
                        # Send to Twilio
                        payload = base64.b64encode(ulaw_data).decode('utf-8')
                        message = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": payload}
                        }
                        await twilio_ws.send(json.dumps(message))

        # Run both tasks concurrently
        await asyncio.gather(
            twilio_to_gemini(),
            gemini_to_twilio_stream()
        )
    """

    print("See code comments for implementation details")
    print()


# Example 4: Configuration for different use cases
def show_configuration_examples():
    """
    Different configurations for various use cases
    """

    print("Configuration Examples")
    print("-" * 40)

    # Standard Gemini to Twilio
    print("1. Gemini Live API → Twilio:")
    print("   AudioStreamConverter(input_rate=24000, output_rate=8000)")
    print()

    # If Gemini changes sample rate in future
    print("2. Gemini 16KHz → Twilio:")
    print("   AudioStreamConverter(input_rate=16000, output_rate=8000)")
    print()

    # For lower latency (smaller chunks)
    print("3. Low latency mode:")
    print("   AudioStreamConverter(")
    print("       input_rate=24000,")
    print("       output_rate=8000,")
    print("       chunk_size=2400  # 100ms chunks")
    print("   )")
    print()

    # Multiple simultaneous calls
    print("4. Multiple calls (create instance per call):")
    print("   converters = {}")
    print("   converters[call_sid] = AudioStreamConverter()")
    print()


# Example 5: Error handling and edge cases
async def robust_streaming_example():
    """
    Example with proper error handling for production
    """

    print("Production-Ready Streaming Example")
    print("-" * 40)

    _converter = AudioStreamConverter()
    _retry_count = 0
    _max_retries = 3

    """
    try:
        while retry_count < max_retries:
            try:
                # Connect to services
                # gemini_ws = await connect_to_gemini()
                # twilio_ws = await connect_to_twilio()

                # Reset converter for new stream
                converter.reset()

                # Stream audio
                # async for gemini_audio in gemini_ws:
                #     try:
                #         # Convert
                #         ulaw_audio = converter.convert_chunk(gemini_audio)
                #
                #         if ulaw_audio:
                #             # Send to Twilio with timeout
                #             await asyncio.wait_for(
                #                 twilio_ws.send(format_for_twilio(ulaw_audio)),
                #                 timeout=5.0
                #             )
                #
                #     except asyncio.TimeoutError:
                #         print("Warning: Twilio send timeout, continuing...")
                #         continue
                #
                #     except Exception as e:
                #         print(f"Error processing chunk: {e}")
                #         # Decide whether to continue or break
                #         continue

                # Success - break retry loop
                break

            except ConnectionError as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Connection error, retry {retry_count}/{max_retries}")
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    raise

    finally:
        # Always flush remaining audio
        try:
            final_audio = converter.flush()
            if final_audio:
                # await send_final_audio(final_audio)
                pass
        except Exception as e:
            print(f"Error flushing final audio: {e}")

        # Clean up connections
        # await cleanup_connections()
    """

    print("See code comments for implementation details")
    print()


async def main():
    """Run all examples"""

    print("=" * 60)
    print("AudioStreamConverter Integration Examples")
    print("=" * 60)
    print()

    await gemini_to_twilio_bridge()
    create_twilio_websocket_server()
    await bidirectional_audio_bridge()
    show_configuration_examples()
    await robust_streaming_example()

    print("=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print()
    print("1. Create one AudioStreamConverter instance per audio stream")
    print("2. Call converter.reset() when starting a new stream")
    print("3. Always call converter.flush() at the end of a stream")
    print("4. Handle variable-sized chunks gracefully")
    print("5. Use asyncio.gather() for bidirectional streaming")
    print("6. Implement proper error handling and retries")
    print("7. Base64 encode μ-law data for Twilio Media Streams")
    print()
    print("For complete working examples, see:")
    print("  - Gemini Live API docs: https://ai.google.dev/")
    print("  - Twilio Media Streams: https://www.twilio.com/docs/voice/media-streams")
    print()


if __name__ == "__main__":
    asyncio.run(main())
