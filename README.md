# VoiceScenarioCoach

VoiceScenarioCoach is an interactive, voice-enabled AI application designed to guide users through workplace scenarios. Built with FastAPI, it simulates real-world situations (e.g., dealing with a micromanaging manager) by processing voice or text input and providing scenario-based coaching with audio responses. The project leverages advanced AI models for speech recognition, natural language processing, and text-to-speech synthesis.

## Features
- **Voice Interaction**: Record audio input via the browser, transcribe it, and receive spoken AI responses.
- **Text Input**: Alternatively, type messages to interact with the AI coach.
- **Scenario-Based Guidance**: Navigate predefined decision points with options, feedback, and best practices.
- **Dynamic UI**: Displays scenario details, conversation history, and clickable options.
- **API-Driven**: FastAPI backend handles audio processing, scenario logic, and response generation.

## Tech Stack
- **Backend**: FastAPI (Python), Uvicorn
- **Frontend**: HTML, CSS, JavaScript (Jinja2 templates)
- **AI/ML**:
  - Speech Recognition: `transformers` (Whisper model)
  - Language Model: `langchain-google-genai` (Google Gemini)
  - Text-to-Speech: `TTS` (Coqui TTS)
- **Audio Processing**: `soundfile`, `pydub`, `numpy`
- **Dependencies**: Managed with `pip`

## Prerequisites
- Python 3.10+
- Google API Key (for Gemini model)
- FFmpeg (for audio format conversion)
- Microphone access in the browser

## Running with Docker

### Prerequisites
- Docker and Docker Compose installed
- docker-compose build
- docker-compose up -d

### Steps
1. **Set Environment Variable**:
   ```bash
   export GOOGLE_API_KEY="your-google-api-key"

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/TanzeelAbbas/Voice-Scenario-AI-Coach.git
   cd Voice-Scenario-AI-Coach