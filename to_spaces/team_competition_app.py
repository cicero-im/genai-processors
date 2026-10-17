import asyncio
import base64
import io
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import PIL.Image
import google.genai as genai
from google.genai import types as genai_types


class LiveQuizTeam:
    def __init__(self):
        self.api_key = None
        self.client = None
        self.connected = False
        self.video_streaming = False
        self.current_question = ""
        self.video_description = "Not connected..."
        self.smart_answer = "Not connected..."
        self.team_final_answer = ""
        self.answer_history = []
        self.question_count = 0
        self.video_session = None
        self.thinking_session = None
        self.stream_task = None
    
    def set_api_key(self, api_key: str):
        """Set and validate the API key"""
        if not api_key:
            return False, "Please provide an API key"
        
        try:
            self.api_key = api_key
            self.client = genai.Client(api_key=api_key)
            return True, "API key set successfully"
        except Exception as e:
            return False, f"Invalid API key: {str(e)}"
    
    def prepare_frame(self, frame):
        """Prepare video frame for Gemini Live API"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame_rgb)
            
            # Resize to thumbnail to reduce token usage
            img.thumbnail([1024, 1024])
            
            # Convert to JPEG and base64 encode
            image_io = io.BytesIO()
            img.save(image_io, format="jpeg")
            image_io.seek(0)
            
            image_bytes = image_io.read()
            return {
                "mime_type": "image/jpeg", 
                "data": base64.b64encode(image_bytes).decode()
            }
        except Exception as e:
            print(f"Frame preparation error: {e}")
            return None

    async def start_video_ai(self):
        """Start the video AI with continuous screen capture streaming"""
        if not self.client:
            return
        
        try:
            # Use the correct Multimodal Live API endpoint
            async with self.client.aio.live.connect(
                model="models/gemini-2.0-flash-exp",
                config={
                    "generation_config": {
                        "response_modalities": ["TEXT"],
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "max_output_tokens": 8192
                    },
                    "system_instruction": {
                        "parts": [{
                            "text": (
                                "You are the EYES of a quiz competition team watching a live video feed. "
                                "CONTINUOUSLY describe everything you see: questions, images, diagrams, text, visual clues. "
                                "When you see a new question or visual content, describe it IMMEDIATELY and COMPLETELY. "
                                "Read all text out loud. Describe all visual elements that could help answer questions. "
                                "This is LIVE competition - your team depends on you seeing everything! "
                                "Format: 'SEEING: [detailed description]' then 'TEXT: [any text you can read]'"
                            )
                        }]
                    }
                }
            ) as session:
                self.video_session = session
                self.video_description = "🎥 Video AI connected, starting screen capture..."
                
                # Start screen capture (using screen instead of camera for quiz content)
                try:
                    import mss
                    sct = mss.mss()
                    monitor = sct.monitors[0]  # Primary monitor
                    
                    # Create tasks for sending frames and receiving responses
                    async def send_frames():
                        while self.video_streaming:
                            try:
                                # Capture screen
                                screenshot = sct.grab(monitor)
                                frame = np.array(screenshot)
                                
                                # Prepare frame for API
                                frame_data = self.prepare_frame(frame)
                                if frame_data:
                                    # Send frame to session
                                    msg = {
                                        "realtime_input": {
                                            "media_chunks": [frame_data]
                                        }
                                    }
                                    await session.send(msg)
                                
                                # Limit frame rate to ~2 FPS to manage tokens
                                await asyncio.sleep(0.5)
                                
                            except Exception as e:
                                print(f"Frame capture error: {e}")
                                await asyncio.sleep(1)
                    
                    async def receive_responses():
                        while self.video_streaming:
                            try:
                                async for response in session.receive():
                                    if response.server_content and response.server_content.model_turn:
                                        if response.server_content.model_turn.parts:
                                            for part in response.server_content.model_turn.parts:
                                                if hasattr(part, 'text') and part.text and self.video_streaming:
                                                    self.video_description = part.text
                                                    print(f"Video AI Update: {part.text}")
                            except Exception as e:
                                print(f"Response processing error: {e}")
                                await asyncio.sleep(1)
                    
                    # Store task for potential cancellation
                    self.stream_task = asyncio.gather(
                        send_frames(),
                        receive_responses()
                    )
                    
                    # Run both tasks concurrently
                    await self.stream_task
                    
                except ImportError:
                    self.video_description = "❌ Screen capture requires 'mss' package. Using text-only mode."
                except Exception as e:
                    self.video_description = f"Screen capture error: {e}"
                        
        except Exception as e:
            print(f"Video AI connection error: {e}")
            self.video_description = f"Video AI Connection Error: {e}. Try restarting."
    
    async def analyze_with_thinking_ai(self, question_text: str):
        """Get analysis from the strategic thinking AI"""
        if not self.client:
            return "No API key set"
        
        try:
            # Use the correct multimodal live API for thinking model
            async with self.client.aio.live.connect(
                model="models/gemini-2.0-flash-thinking-exp",
                config={
                    "generation_config": {
                        "response_modalities": ["TEXT"],
                        "temperature": 0.2,
                        "max_output_tokens": 4096
                    },
                    "system_instruction": {
                        "parts": [{
                            "text": (
                                "You are the Strategic AI for a live quiz team. "
                                "Analyze questions and visual context quickly and provide smart answers. "
                                "Give a confident answer immediately - speed matters in live competition! "
                                "Format: 'ANSWER: [your answer]' followed by brief reasoning."
                            )
                        }]
                    }
                }
            ) as session:
                
                # Send context and question
                context = f"Video AI sees: {self.video_description}\nQuestion: {question_text}"
                
                await session.send(context, end_of_turn=True)
                
                response_text = ""
                async for response in session.receive():
                    if response.server_content and response.server_content.model_turn:
                        if response.server_content.model_turn.parts:
                            for part in response.server_content.model_turn.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text += part.text
                
                self.smart_answer = response_text
                return response_text
                
        except Exception as e:
            error_msg = f"Thinking AI Error: {e}"
            self.smart_answer = error_msg
            return error_msg


# Global team instance
team = LiveQuizTeam()


def connect_team(api_key: str):
    """Connect to Gemini API"""
    success, message = team.set_api_key(api_key)
    if success:
        team.connected = True
        team.video_description = "🔗 Connected to Gemini API. Ready to start video streaming."
        team.smart_answer = "🔗 Connected to Gemini API. Ready for analysis."
        return "✅ Connected to Gemini API! Now you can start video streaming."
    else:
        return f"❌ {message}"


def start_video_streaming():
    """Start video streaming"""
    if not team.connected:
        return "❌ Connect to API first!"
    
    if team.video_streaming:
        return "⚠️ Video streaming already active!"
    
    team.video_streaming = True
    
    # Start video AI in background thread
    def run_video_ai():
        asyncio.run(team.start_video_ai())
    
    threading.Thread(target=run_video_ai, daemon=True).start()
    return "🎥 Video streaming started! AI is now watching your screen."


def stop_video_streaming():
    """Stop video streaming"""
    team.video_streaming = False
    if team.stream_task:
        team.stream_task.cancel()
    team.video_description = "⏹️ Video streaming stopped."
    return "⏹️ Video streaming stopped."


def disconnect_team():
    """Disconnect from API"""
    team.connected = False
    team.video_streaming = False
    if team.stream_task:
        team.stream_task.cancel()
    team.video_description = "Disconnected..."
    team.smart_answer = "Disconnected..."
    return "🔌 Disconnected from API."


def quick_analyze(question_text: str = ""):
    """Quickly analyze current question"""
    if not team.client:
        return "❌ Set API key first!", team.video_description, "Set API key first"
    
    if question_text:
        team.current_question = question_text
    
    # Get AI analysis (run in thread to avoid blocking)
    def get_analysis():
        return asyncio.run(team.analyze_with_thinking_ai(team.current_question or question_text))
    
    try:
        thinking_response = get_analysis()
    except Exception as e:
        thinking_response = f"Error: {e}"
    
    return (
        f"🧠 Analyzing: {team.current_question or 'Current screen content'}",
        team.video_description,
        thinking_response
    )


def submit_team_answer(answer: str):
    """Submit your team's final answer"""
    if not team.client:
        return "❌ Set API key first!"
    
    if not answer.strip():
        return "❌ Please enter an answer"
    
    team.team_final_answer = answer
    team.question_count += 1
    
    # Record this answer
    team.answer_history.append({
        'question_num': team.question_count,
        'question': team.current_question,
        'video_saw': team.video_description[-100:],
        'ai_suggested': team.smart_answer[-100:],
        'team_answer': answer,
        'timestamp': time.time()
    })
    
    return f"✅ ANSWER SUBMITTED: {answer} (Question #{team.question_count})"


def get_team_status():
    """Get current team status"""
    if not team.client:
        return "Not started", "No answers yet", team.video_description, team.smart_answer
    
    return (
        f"Questions: {team.question_count}",
        f"Last: {team.team_final_answer}",
        team.video_description,
        team.smart_answer
    )


def get_answer_history():
    """Get formatted answer history"""
    if not team.answer_history:
        return "No answers submitted yet"
    
    history = "📋 ANSWER HISTORY:\n\n"
    for entry in team.answer_history[-5:]:  # Show last 5
        history += f"Q{entry['question_num']}: {entry['team_answer']}\n"
        history += f"   Question: {entry['question'][:50]}...\n"
        history += f"   Time: {time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))}\n\n"
    
    return history


# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="Live Quiz Competition - Team Command Center",
        theme=gr.themes.Soft(),
        css="""
        .primary-btn { background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important; }
        .video-feed { border: 3px solid #4ECDC4; }
        .thinking-feed { border: 3px solid #FFE66D; }
        """
    ) as interface:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎥 LIVE Quiz Competition - Team Command Center</h1>
            <h2>Your Team: You + Video AI + Strategic Thinking AI</h2>
            <p><strong>Video AI watches your screen/video feed • Strategic AI analyzes and answers</strong></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Connection Setup")
                api_key_input = gr.Textbox(
                    label="Google API Key", 
                    type="password",
                    placeholder="Enter your Google API key here..."
                )
                connect_btn = gr.Button(
                    "🔗 Connect to API", 
                    variant="primary", 
                    size="lg"
                )
                disconnect_btn = gr.Button("🔌 Disconnect", variant="secondary")
                connection_status = gr.Textbox(label="Connection Status", interactive=False)
                
            with gr.Column(scale=1):
                gr.Markdown("## 🎥 Video Controls")
                start_video_btn = gr.Button(
                    "▶️ Start Video Stream", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["primary-btn"]
                )
                stop_video_btn = gr.Button("⏹️ Stop Video Stream", variant="secondary")
                video_status = gr.Textbox(label="Video Status", interactive=False)
                
            with gr.Column(scale=2):
                gr.Markdown("## 🎯 Quick Actions")
                quick_question_input = gr.Textbox(
                    label="Heard a question? Type it here for AI analysis", 
                    lines=2,
                    placeholder="Type the question you heard from the mediator..."
                )
                analyze_btn = gr.Button(
                    "🧠 Analyze Current Screen + Question", 
                    variant="primary",
                    size="lg"
                )
                question_status = gr.Textbox(label="Analysis Status", interactive=False)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 👁️ VIDEO AI FEED")
                gr.Markdown("*This AI watches your screen/video and describes everything it sees*")
                video_output = gr.Textbox(
                    label="🎥 What Video AI Sees RIGHT NOW", 
                    lines=8, 
                    interactive=False,
                    elem_classes=["video-feed"]
                )
                
            with gr.Column(scale=1):
                gr.Markdown("## 🧠 STRATEGIC AI ANALYSIS")
                gr.Markdown("*This AI analyzes questions and provides smart answers*")
                thinking_output = gr.Textbox(
                    label="💡 AI Strategic Analysis & Answer", 
                    lines=8, 
                    interactive=False,
                    elem_classes=["thinking-feed"]
                )
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 🎯 YOUR TEAM'S FINAL ANSWER")
                team_answer_input = gr.Textbox(
                    label="Team's Final Answer (You + AIs)", 
                    lines=3,
                    placeholder="Combine your judgment with AI suggestions and type final answer..."
                )
                submit_final_btn = gr.Button(
                    "📤 SUBMIT TEAM ANSWER", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["primary-btn"]
                )
                answer_status = gr.Textbox(label="Submission Status", interactive=False)
                
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Competition Status")
                questions_answered = gr.Textbox(label="Questions Answered", interactive=False)
                last_answer = gr.Textbox(label="Last Submitted Answer", interactive=False)
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Status")
                    history_btn = gr.Button("📋 History")
                
                answer_history_display = gr.Textbox(
                    label="Recent Answers", 
                    lines=6, 
                    interactive=False
                )
        
        # Add auto-refresh for live feeds
        def auto_refresh():
            return get_team_status()
        
        # Wire up the interface
        connect_btn.click(
            connect_team,
            inputs=[api_key_input],
            outputs=[connection_status]
        )
        
        disconnect_btn.click(
            disconnect_team,
            outputs=[connection_status]
        )
        
        start_video_btn.click(
            start_video_streaming,
            outputs=[video_status]
        )
        
        stop_video_btn.click(
            stop_video_streaming,
            outputs=[video_status]
        )
        
        analyze_btn.click(
            quick_analyze,
            inputs=[quick_question_input],
            outputs=[question_status, video_output, thinking_output]
        )
        
        submit_final_btn.click(
            submit_team_answer,
            inputs=[team_answer_input],
            outputs=[answer_status]
        )
        
        refresh_btn.click(
            get_team_status,
            outputs=[questions_answered, last_answer, video_output, thinking_output]
        )
        
        history_btn.click(
            get_answer_history,
            outputs=[answer_history_display]
        )
        
        # Manual refresh on load
        interface.load(
            get_team_status,
            outputs=[questions_answered, last_answer, video_output, thinking_output]
        )
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #eee;">
            <p><strong>How to use:</strong></p>
            <p>1. Enter your Google API key and click "🔗 Connect to API"</p>
            <p>2. Click "▶️ Start Video Stream" to begin screen monitoring</p>
            <p>3. When you hear a question, type it in and click analyze</p>
            <p>4. Review both AI feeds and submit your team's final answer</p>
            <p>5. Use "⏹️ Stop Video Stream" to pause monitoring</p>
            <p><em>Perfect for live Zoom quiz competitions!</em></p>
        </div>
        """)
    
    return interface


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Share the Gradio interface")
    args = parser.parse_args()
    
    interface = create_interface()
    interface.launch(share=args.share)