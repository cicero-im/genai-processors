import argparse
import asyncio
import os

import google.genai as genai
from google.genai import types as genai_types
from genai_processors import content_api
from genai_processors.core import audio_io
import pyaudio


def _get_api_key() -> str:
    """Get the Google API key from env or prompt."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google API key: ").strip()
        os.environ["GOOGLE_API_KEY"] = api_key
    return api_key


async def play_music(prompt: str, model_name: str, sample_rate: int) -> None:
    """Connects to the live music API and plays the generated audio."""
    client = genai.Client(api_key=_get_api_key())
    pya = pyaudio.PyAudio()
    try:
        output = audio_io.PyAudioOut(pya, rate=sample_rate)

        async with client.aio.live.music.connect(model=model_name) as session:
            await session.set_weighted_prompts(
                [genai_types.WeightedPrompt(text=prompt, weight=1.0)]
            )
            await session.play()

            async def to_audio_parts():
                async for msg in session.receive():
                    if msg.server_content and msg.server_content.audio_chunks:
                        for chunk in msg.server_content.audio_chunks:
                            yield content_api.ProcessorPart(
                                genai_types.Part.from_bytes(
                                    data=chunk.data,
                                    mime_type=chunk.mime_type or "audio/pcm",
                                ),
                                role="ASSISTANT",
                            )

            async for _ in output(to_audio_parts()):
                pass
    finally:
        pya.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music using the Live Music API")
    parser.add_argument("--prompt", default="a pleasant melody", help="Music prompt")
    parser.add_argument("--model", default="models/music-bison-001", help="Model name")
    parser.add_argument("--rate", type=int, default=24000, help="Sample rate for playback")
    args = parser.parse_args()
    asyncio.run(play_music(args.prompt, args.model, args.rate))
