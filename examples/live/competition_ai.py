import argparse
import asyncio
import os


def _get_api_key() -> str:
    """Returns the API key, prompting the user if necessary."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google API key: ").strip()
        os.environ["GOOGLE_API_KEY"] = api_key
    return api_key

from genai_processors import content_api
from genai_processors.core import audio_io
from genai_processors.core import video
import commentator
import pyaudio

# Custom prompt instructing the model to act as a quiz competitor.
QUIZ_PROMPT_PARTS = [
    (
        "You are a contestant in a quiz competition. Answer the questions on "
        "the screen or from the host as quickly and accurately as possible."
    ),
    (
        "Keep your answers short and direct. Do not add commentary unless "
        "asked."
    ),
]

API_KEY = _get_api_key()


async def run_competition(video_mode: str) -> None:
    """Runs the AI competition participant."""
    pya = pyaudio.PyAudio()
    try:
        video_mode_enum = video.VideoMode(video_mode)
        input_processor = video.VideoIn(video_mode=video_mode_enum) + audio_io.PyAudioIn(pya)

        async def input_stream():
            try:
                while True:
                    await asyncio.sleep(1)
            finally:
                yield content_api.ProcessorPart("Ending the stream")

        competitor_processor = commentator.create_live_commentator(
            API_KEY,
            chattiness=0.0,
            prompt_parts=QUIZ_PROMPT_PARTS,
        )

        consume_output = audio_io.PyAudioOut(pya)

        agent = input_processor + competitor_processor + consume_output

        async for _ in agent(input_stream()):
            pass
    finally:
        pya.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="screen",
        help="Video source",
        choices=["camera", "screen"],
    )
    args = parser.parse_args()
    asyncio.run(run_competition(video_mode=args.mode))
