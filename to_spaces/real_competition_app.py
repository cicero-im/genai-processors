import asyncio
import base64
import io
import time
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import PIL.Image
import google.genai as genai
from google.genai import types as genai_types
from gradio_webrtc import WebRTC


class GeminiQuizHandler:
    """Handler for real-time video analysis using Gemini Live API"""

    def __init__(self):
        self.client = None
        self.session = None
        self.quit = asyncio.Event()
        self.connected = False
        self.api_key = None
        self.question_count = 0
        self.answers = []
        self.last_description = "Waiting for connection..."
        self.last_analysis = "Ready for questions..."

    def copy(self):
        """Create a new handler instance for each user"""
        return GeminiQuizHandler()

    async def connect(self, api_key: str):
        """Connect to Gemini Live API"""
        if not api_key:
            raise ValueError("API key required")

        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)

        try:
            # Test connection
            await self.client.aio.models.get("models/gemini-2.0-flash-exp")
            self.connected = True
            return True
        except Exception as e:
            raise ValueError(f"Failed to connect: {str(e)}")

    async def start_session(self):
        """Start Live API session for video analysis"""
        if not self.connected:
            return

        try:
            async with self.client.aio.live.connect(
                model="models/gemini-2.0-flash-exp",
                config={
                    "generation_config": {
                        "response_modalities": ["TEXT"],
                        "temperature": 0.1,
                        "max_output_tokens": 4096
                    },
                    "system_instruction": {
                        "parts": [{
                            "text": (
                                "You are the VIDEO AI for a live quiz competition team. "
                                "Analyze everything you see: questions, images, diagrams, text, visual clues. "
                                "When you see quiz content, describe it immediately and completely. "
                                "Read all text you can see. Identify visual elements that help answer questions. "
                                "Be quick and precise - this is live competition! "
                                "Format: 'SEEING: [description]' then 'TEXT: [any readable text]'"
                            )
                        }]
                    }
                }
            ) as session:
                self.session = session

                # Keep session alive until quit signal
                async for response in session.receive():
                    if self.quit.is_set():
                        break

                    if response.server_content and response.server_content.model_turn:
                        if response.server_content.model_turn.parts:
                            for part in response.server_content.model_turn.parts:
                                if hasattr(part, 'text') and part.text:
                                    self.last_description = part.text
                                    print(f"Video AI: {part.text}")

        except Exception as e:
            print(f"Session error: {e}")
            self.last_description = f"Session error: {e}"

    async def analyze_question(self, question_text: str):
        """Analyze a question using thinking model"""
        if not self.connected:
            return "Not connected to API"

        try:
            async with self.client.aio.live.connect(
                model="models/gemini-2.0-flash-exp",
                config={
                    "generation_config": {
                        "response_modalities": ["TEXT"],
                        "temperature": 0.2,
                        "max_output_tokens": 2048
                    },
                    "system_instruction": {
                        "parts": [{
                            "text": (
                                "You are the STRATEGIC AI for a quiz team. "
                                "Analyze questions quickly and provide smart answers. "
                                "Consider the visual context and give confident responses. "
                                "Format: 'ANSWER: [your answer]' then brief reasoning."
                            )
                        }]
                    }
                }
            ) as session:

                context = f"Video context: {self.last_description}\nQuestion: {question_text}"
                await session.send(context, end_of_turn=True)

                response_text = ""
                async for response in session.receive():
                    if response.server_content and response.server_content.model_turn:
                        if response.server_content.model_turn.parts:
                            for part in response.server_content.model_turn.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text += part.text

                self.last_analysis = response_text
                return response_text

        except Exception as e:
            error_msg = f"Analysis error: {e}"
            self.last_analysis = error_msg
            return error_msg

    def prepare_frame(self, frame):
        """Convert video frame for Gemini API"""
        try:
            # Convert to RGB and resize
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            img = PIL.Image.fromarray(frame_rgb)
            img.thumbnail([1024, 1024])

            # Convert to JPEG and base64
            image_io = io.BytesIO()
            img.save(image_io, format="jpeg")
            image_io.seek(0)

            image_bytes = image_io.read()
            return {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_bytes).decode()
            }
        except Exception as e:
            print(f"Frame prep error: {e}")
            return None

    async def process_frame(self, frame):
        """Process incoming video frame"""
        if not self.session or self.quit.is_set():
            return frame

        try:
            # Prepare and send frame
            frame_data = self.prepare_frame(frame)
            if frame_data:
                msg = {
                    "realtime_input": {
                        "media_chunks": [frame_data]
                    }
                }
                await self.session.send(msg)

        except Exception as e:
            print(f"Frame processing error: {e}")

        return frame

    async def shutdown(self):
        """Shutdown the handler"""
        self.quit.set()
        if self.session:
            try:
                await self.session.close()
            except:
                pass


# Global handler instance
quiz_handler = GeminiQuizHandler()


def create_interface():
    """Create the real-time quiz competition interface"""

    with gr.Blocks(
        title="Live Quiz Competition - Real-Time Video AI",
        theme=gr.themes.Soft()
    ) as interface:

        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎥 Live Quiz Competition - Real-Time Video AI</h1>
            <h2>Your AI Team: Video AI + Strategic AI + You</h2>
            <p><strong>Real-time video analysis with WebRTC streaming</strong></p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🔗 API Setup")
                api_key_input = gr.Textbox(
                    label="Google API Key",
                    type="password",
                    placeholder="Enter your API key..."
                )
                connect_btn = gr.Button("Connect to API", variant="primary")
                connection_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("## 🎥 Live Video Stream")
                webrtc = WebRTC(
                    label="Video Stream",
                    modality="video",
                    mode="send-receive",
                    elem_id="video-source"
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("## 👁️ Video AI Analysis")
                video_output = gr.Textbox(
                    label="What Video AI Sees",
                    lines=6,
                    interactive=False,
                    value="Connect API and start video to begin..."
                )

            with gr.Column():
                gr.Markdown("## 🧠 Strategic Analysis")
                question_input = gr.Textbox(
                    label="Question from mediator",
                    lines=2,
                    placeholder="Type the question you heard..."
                )
                analyze_btn = gr.Button("Analyze Question", variant="primary")
                strategy_output = gr.Textbox(
                    label="Strategic AI Answer",
                    lines=6,
                    interactive=False
                )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 🎯 Team Answer")
                team_answer_input = gr.Textbox(
                    label="Your Team's Final Answer",
                    lines=3,
                    placeholder="Combine AI insights with your judgment..."
                )
                submit_btn = gr.Button("Submit Answer", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("## 📊 Status")
                answer_status = gr.Textbox(label="Submission Status", interactive=False)
                question_counter = gr.Textbox(label="Questions Answered", interactive=False)

        # Connection logic
        async def connect_api(api_key):
            try:
                await quiz_handler.connect(api_key)
                # Start the session in background
                asyncio.create_task(quiz_handler.start_session())
                return "✅ Connected! Start video stream to begin analysis."
            except Exception as e:
                return f"❌ Connection failed: {str(e)}"

        # Analysis logic
        async def analyze_question_async(question):
            if not question.strip():
                return "Please enter a question"

            result = await quiz_handler.analyze_question(question)
            return result

        # Answer submission
        def submit_answer(answer):
            if not answer.strip():
                return "Please enter an answer", quiz_handler.question_count

            quiz_handler.question_count += 1
            quiz_handler.answers.append({
                'number': quiz_handler.question_count,
                'answer': answer,
                'timestamp': time.time()
            })

            return f"✅ Answer #{quiz_handler.question_count} submitted: {answer}", quiz_handler.question_count

        # Status updates
        def update_status():
            return quiz_handler.last_description, quiz_handler.last_analysis

        # Wire up the interface
        connect_btn.click(
            connect_api,
            inputs=[api_key_input],
            outputs=[connection_status]
        )

        analyze_btn.click(
            analyze_question_async,
            inputs=[question_input],
            outputs=[strategy_output]
        )

        submit_btn.click(
            submit_answer,
            inputs=[team_answer_input],
            outputs=[answer_status, question_counter]
        )

        # Set up real-time video streaming
        webrtc.stream(
            quiz_handler.process_frame,
            inputs=[webrtc],
            outputs=[webrtc],
            time_limit=120,
            concurrency_limit=3
        )

        # Auto-refresh status every 2 seconds
        interface.load(
            update_status,
            outputs=[video_output, strategy_output]
        )

        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #eee;">
            <p><strong>Instructions:</strong></p>
            <p>1. Enter API key and connect</p>
            <p>2. Allow camera/screen access for video stream</p>
            <p>3. Point camera at quiz content or share screen</p>
            <p>4. Type questions and get AI analysis</p>
            <p>5. Submit your team's final answers</p>
        </div>
        """)

    return interface


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Share the interface")
    args = parser.parse_args()

    interface = create_interface()
    interface.launch(share=args.share)
