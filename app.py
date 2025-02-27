from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import threading
import tempfile
import time
import numpy as np
import json
import base64
import soundfile as sf
from pydub import AudioSegment
from voice_ai_agent import ScenarioVoiceAI

app = FastAPI(
    title="Scenario Voice AI API",
    description="API for interacting with a workplace scenario voice AI agent."
)
agent = None
agent_lock = threading.Lock()

# Directory for temporary files
TEMP_DIR = os.path.join(tempfile.gettempdir(), "voice_ai_agent")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Get the voice AI agent instance (singleton pattern)
def get_agent():
    global agent
    with agent_lock:
        if agent is None:
            print("Initializing Voice AI Agent...")
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            agent = ScenarioVoiceAI(api_key=api_key)
        return agent

# Serve the main HTML page
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Model for text processing
class TextRequest(BaseModel):
    text: str

# New endpoint for initial message with audio
@app.get("/api/initial-message")
async def get_initial_message():
    try:
        voice_agent = get_agent()
        intro = f"Welcome to the scenario: {voice_agent.scenario['title']}. {voice_agent.scenario['background']} Let's start with the initial situation: {voice_agent.scenario['initial_situation']} Would you like me to present the first decision point?"
        temp_path = os.path.join(TEMP_DIR, f"initial_{time.time()}.wav")
        voice_agent._speak_to_file(intro, temp_path)
        with open(temp_path, 'rb') as f:
            audio_data = f.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        os.remove(temp_path)
        return {"response": intro, "audio": audio_base64}
    except Exception as e:
        print(f"Error generating initial message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Process voice input
@app.post("/api/process-voice")
async def process_voice(audio: UploadFile = File(...)):
    try:
        # Save the uploaded WebM audio file temporarily
        temp_webm_path = os.path.join(TEMP_DIR, f"input_{time.time()}.webm")
        with open(temp_webm_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        # Convert WebM to WAV
        temp_wav_path = os.path.join(TEMP_DIR, f"input_{time.time()}.wav")
        audio_segment = AudioSegment.from_file(temp_webm_path, format="webm")
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)  # Match ScenarioVoiceAI sample_rate
        audio_segment.export(temp_wav_path, format="wav")
        
        # Get the agent instance
        voice_agent = get_agent()
        
        # Process the audio file
        audio_data, sample_rate = sf.read(temp_wav_path)
        if sample_rate != voice_agent.sample_rate:
            raise HTTPException(status_code=400, detail=f"Audio sample rate must be {voice_agent.sample_rate} Hz")
        
        # Convert to the format expected by the agent
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)  # Convert stereo to mono
        
        # Get transcription
        text = voice_agent._speech_to_text(audio_data)
        
        # Clean up temporary files
        os.remove(temp_webm_path)
        os.remove(temp_wav_path)
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        # Generate response
        response = voice_agent._generate_response(text)
        voice_agent.conversation_history.append({"role": "user", "content": text})
        voice_agent.conversation_history.append({"role": "assistant", "content": response})
        
        # Generate speech for the response
        temp_output_path = os.path.join(TEMP_DIR, f"output_{time.time()}.wav")
        voice_agent._speak_to_file(response, temp_output_path)
        
        # Read the audio file and convert to base64
        with open(temp_output_path, 'rb') as f:
            audio_data = f.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Clean up temporary file
        os.remove(temp_output_path)
        
        return {
            "text": text,
            "response": response,
            "audio": audio_base64
        }
    
    except Exception as e:
        print(f"Error processing voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Process text input and get AI response
@app.post("/api/process-text")
async def process_text(request: TextRequest):
    try:
        user_text = request.text
        
        # Get the agent instance
        voice_agent = get_agent()
        
        # Generate response
        response = voice_agent._generate_response(user_text)
        voice_agent.conversation_history.append({"role": "user", "content": user_text})
        voice_agent.conversation_history.append({"role": "assistant", "content": response})
        
        # Generate speech for the response
        temp_path = os.path.join(TEMP_DIR, f"output_{time.time()}.wav")
        voice_agent._speak_to_file(response, temp_path)
        
        # Read the audio file and convert to base64
        with open(temp_path, 'rb') as f:
            audio_data = f.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return {
            "text": user_text,
            "response": response,
            "audio": audio_base64
        }
    
    except Exception as e:
        print(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get the scenario data
@app.get("/api/scenario")
async def get_scenario():
    try:
        voice_agent = get_agent()
        return voice_agent.scenario
    except Exception as e:
        print(f"Error retrieving scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get conversation history
@app.get("/api/history")
async def get_history():
    try:
        voice_agent = get_agent()
        return {"conversation_history": voice_agent.conversation_history}
    except Exception as e:
        print(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a method to ScenarioVoiceAI to save speech to a file instead of playing it
def speak_to_file(self, text, file_path):
    try:
        self.tts.tts_to_file(text=text, file_path=file_path, speaker="p339")
        return True
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return False

# Add the method to the ScenarioVoiceAI class
ScenarioVoiceAI._speak_to_file = speak_to_file

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)