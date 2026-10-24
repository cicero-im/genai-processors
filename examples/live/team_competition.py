#!/usr/bin/env python3
"""
AI Team Competition App - Live Zoom Competition
Your team: You + Video AI + Smart Thinking Model vs Friend's team
Mediator asks questions, teams answer quickly, scores tracked later
"""
import argparse
import asyncio
import os
import threading
import time
from typing import Optional

import gradio as gr
import pyaudio

from genai_processors import content_api
from genai_processors.core import audio_io, video
import commentator


def _get_api_key() -> str:
    """Returns the API key, prompting the user if necessary."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google API key: ").strip()
        os.environ["GOOGLE_API_KEY"] = api_key
    return api_key


class TeamCompetition:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.pya = pyaudio.PyAudio()
        self.running = False
        self.current_question = ""
        self.video_description = "Waiting for video input..."
        self.smart_answer = "Waiting for question..."
        self.team_final_answer = ""
        self.answer_history = []
        self.question_count = 0
        
        # Video AI - PRIMARY input source, watches Zoom/screen continuously  
        self.video_ai = commentator.create_live_commentator(
            api_key,
            chattiness=1.0,
            prompt_parts=[
                "You are the EYES of a quiz competition team. The mediator is showing content on video/screen.",
                "IMMEDIATELY describe everything you see: questions, images, diagrams, text, visual clues, charts.",
                "When the mediator shows a new question or visual, describe it COMPLETELY and QUICKLY.",
                "Read all text out loud. Describe all visual elements that could be quiz answers.",
                "This is LIVE competition - your team depends on you seeing everything!",
                "Format: 'SEEING: [detailed description]' then 'TEXT: [any text you can read]'",
            ],
            model_name="models/gemini-2.0-flash-exp"
        )
        
        # Smart thinking model - analyzes questions quickly
        self.thinking_ai = commentator.create_live_commentator(
            api_key,
            chattiness=0.9,
            prompt_parts=[
                "You are the Strategic AI for a live quiz team. When given a question, provide a quick, confident answer.",
                "Be fast but accurate. Don't overthink - give your best answer immediately.",
                "Format: 'ANSWER: [your answer]' followed by brief reasoning if time allows.",
                "This is a live competition - speed matters!",
            ],
            model_name="models/gemini-2.0-flash-thinking-exp"
        )
    
    async def start_video_ai(self):
        """Start the video AI that watches the screen"""
        try:
            video_mode = video.VideoMode("screen")
            input_processor = video.VideoIn(video_mode=video_mode) + audio_io.PyAudioIn(self.pya)
            
            async def input_stream():
                try:
                    while self.running:
                        await asyncio.sleep(0.1)
                finally:
                    yield content_api.ProcessorPart("Video AI ending")
            
            async def capture_video_output(stream):
                async for part in stream:
                    if hasattr(part, 'text') and part.text:
                        self.video_description = part.text
                        print(f"\n{part.text}")
            
            video_agent = input_processor + self.video_ai
            await capture_video_output(video_agent(input_stream()))
            
        except Exception as e:
            print(f"Video AI error: {e}")
        finally:
            self.pya.terminate()
    
    def send_to_thinking_ai(self, question_text: str) -> str:
        """Send question to thinking AI for analysis"""
        try:
            # Create a simple text input for the thinking AI
            prompt = f"Question: {question_text}\nContext: {self.video_description}"
            
            # For now, return a placeholder - in a full implementation,
            # you'd process this through the thinking AI
            return f"THINKING AI SUGGESTS: Analyzing question: {question_text[:100]}..."
            
        except Exception as e:
            return f"THINKING AI ERROR: {e}"


def create_competition_interface():
    """Create Gradio interface for the competition"""
    
    competition = None
    
    def start_competition(api_key: str):
        nonlocal competition
        if not api_key:
            return "Please provide an API key"
        
        competition = TeamCompetition(api_key)
        competition.running = True
        
        # Start video AI in background thread
        def run_video_ai():
            asyncio.run(competition.start_video_ai())
        
        threading.Thread(target=run_video_ai, daemon=True).start()
        return "Competition started! Video AI is watching your screen."
    
    def stop_competition():
        nonlocal competition
        if competition:
            competition.running = False
        return "Competition stopped."
    
    def quick_analyze(question_text: str = ""):
        """Quickly analyze current question - used when mediator asks something"""
        nonlocal competition
        if not competition:
            return "Start competition first!", "", ""
        
        if question_text:
            competition.current_question = question_text
        
        # Get AI analysis
        thinking_response = competition.send_to_thinking_ai(competition.current_question or question_text)
        
        return (
            f"Analyzing: {competition.current_question or 'Current screen content'}",
            competition.video_description,
            thinking_response
        )
    
    def submit_team_answer(answer: str):
        """Submit your team's final answer for this question"""
        nonlocal competition
        if not competition:
            return "Start competition first!"
        
        competition.team_final_answer = answer
        competition.question_count += 1
        
        # Record this answer
        competition.answer_history.append({
            'question_num': competition.question_count,
            'question': competition.current_question,
            'video_saw': competition.video_description[-100:],  # Last 100 chars
            'ai_suggested': competition.smart_answer[-100:],    # Last 100 chars  
            'team_answer': answer,
            'timestamp': time.time()
        })
        
        return f"✅ ANSWER SUBMITTED: {answer} (Question #{competition.question_count})"
    
    def get_team_status():
        nonlocal competition
        if not competition:
            return "Competition not started", ""
        
        return (
            f"Questions Answered: {competition.question_count}",
            f"Last Answer: {competition.team_final_answer}"
        )
    
    # Create Gradio interface
    with gr.Blocks(title="Live Quiz Competition - Team Command Center") as interface:
        gr.Markdown("# 🎥 LIVE Quiz Competition - Your Team Command Center")
        gr.Markdown("**Your Team:** You + Video AI (watching Zoom/screen) + Strategic Thinking AI")
        gr.Markdown("*The Video AI is your primary input - it watches the mediator's screen/video feed*")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Setup")
                api_key_input = gr.Textbox(label="Google API Key", type="password")
                start_btn = gr.Button("🚀 Start Watching Video", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                status_output = gr.Textbox(label="System Status", interactive=False)
                
            with gr.Column(scale=2):
                gr.Markdown("## 🎯 Quick Actions")
                quick_question_input = gr.Textbox(label="Heard a question? Type it here for AI analysis", lines=2)
                analyze_btn = gr.Button("🧠 Analyze Current Screen + Question", variant="primary")
                question_status = gr.Textbox(label="Analysis Status", interactive=False)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 👁️ VIDEO AI FEED (MAIN INPUT)")
                video_output = gr.Textbox(label="🎥 What Video AI Sees RIGHT NOW", lines=8, interactive=False)
                
            with gr.Column(scale=2):
                gr.Markdown("## 🧠 STRATEGIC AI ANALYSIS")
                thinking_output = gr.Textbox(label="💡 AI Strategic Analysis & Answer", lines=8, interactive=False)
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## 🎯 YOUR TEAM'S FINAL ANSWER")
                team_answer_input = gr.Textbox(label="Team's Final Answer (You + AIs)", lines=3, placeholder="Type your team's answer here...")
                submit_final_btn = gr.Button("📤 SUBMIT TEAM ANSWER", variant="primary", size="lg")
                answer_status = gr.Textbox(label="Submission Status", interactive=False)
                
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Competition Status")
                questions_answered = gr.Textbox(label="Questions Answered", interactive=False)
                last_answer = gr.Textbox(label="Last Submitted Answer", interactive=False)
                
                refresh_btn = gr.Button("🔄 Refresh Status")
                answer_history_btn = gr.Button("📋 View Answer History")
        
        # Wire up the interface
        start_btn.click(
            start_competition,
            inputs=[api_key_input],
            outputs=[status_output]
        )
        
        stop_btn.click(
            stop_competition,
            outputs=[status_output]
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
            outputs=[questions_answered, last_answer]
        )
    
    return interface


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Share the Gradio interface")
    args = parser.parse_args()
    
    interface = create_competition_interface()
    interface.launch(share=args.share)