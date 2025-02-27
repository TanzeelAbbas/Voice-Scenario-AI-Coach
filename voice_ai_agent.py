import os
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from transformers import pipeline
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScenarioVoiceAI:
    def __init__(self, scenario_input, api_key=None):
        self.running = False
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        self.listening_thread = None
        self.speaking = False
        self.recording = False
        self.conversation_history = []
        self.current_decision_point = None
        
        # Check for API key
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key must be provided either directly or via GOOGLE_API_KEY environment variable")
        
        # Initialize language model first
        logging.info("Initializing Google Gemini language model via Langchain...")
        self.language_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024,
        )
        
        # Process user-provided scenario input
        self.scenario_text = scenario_input.strip()
        if not self.scenario_text:
            raise ValueError("Scenario input cannot be empty")
        logging.info(f"Received scenario input: {self.scenario_text}")
        self.scenario = self._interpret_scenario(self.scenario_text)
        
        logging.info("Initializing speech recognition model...")
        self.speech_recognition = pipeline("automatic-speech-recognition", model="openai/whisper-base")
        
        logging.info("Initializing text-to-speech model...")
        self.tts = TTS("tts_models/en/vctk/vits") 
        
        # Audio recording parameters
        self.sample_rate = 16000
        self.recording_buffer = []
        self.silence_threshold = 0.02
        self.silence_duration = 1.5
        self.min_speech_duration = 1.0
        self.last_sound = time.time()
        self.speech_start_time = None
        
        logging.info("Scenario Voice AI initialized with user-provided scenario!")

    def _interpret_scenario(self, scenario_text):
        """Interpret user input into a flexible scenario structure."""
        try:
            # Try parsing as JSON first
            scenario = json.loads(scenario_text)
            logging.info("Input parsed as JSON successfully")
            # Minimal validation: Ensure it’s a dict with some content
            if not isinstance(scenario, dict) or not scenario:
                raise ValueError("JSON input must be a non-empty dictionary")
            # Add defaults if key fields are missing
            if "title" not in scenario:
                scenario["title"] = "Custom Scenario"
            if "background" not in scenario:
                scenario["background"] = scenario_text[:100] + "..." if len(scenario_text) > 100 else scenario_text
            if "initial_situation" not in scenario:
                scenario["initial_situation"] = "User-defined situation begins here."
            if "decision_points" not in scenario or not isinstance(scenario["decision_points"], list) or not scenario["decision_points"]:
                scenario["decision_points"] = [{
                    "id": "dp1",
                    "name": "Initial Decision",
                    "situation": "Decide how to proceed based on the scenario.",
                    "options": [
                        {"id": "A", "text": "Take action", "implications": "Proactive approach"},
                        {"id": "B", "text": "Wait and see", "implications": "Less immediate effort"}
                    ],
                    "best_practice": "A",
                    "best_practice_explanation": "Addresses the issue proactively."
                }]
            return scenario
        except json.JSONDecodeError:
            # Interpret paragraph or invalid JSON with AI
            logging.info("Input is not JSON, interpreting as paragraph...")
            prompt = f"""
            Interpret this user-provided scenario text into a structured format for a workplace coaching AI:
            "{scenario_text}"
            Generate a JSON object with:
            - title: A concise title based on the text
            - background: A brief context summary
            - initial_situation: The starting situation
            - decision_points: At least one decision point with id, name, situation, options (id, text, implications), best_practice, and best_practice_explanation
            If the text is unclear, make reasonable assumptions to create a usable scenario.
            Return the JSON object as a valid JSON string. If you cannot generate a valid scenario, return a default JSON structure instead.
            """
            try:
                response = self.language_model.invoke(prompt).content
                logging.info(f"Raw model response: {response}")
                # Attempt to extract JSON if embedded in text
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    scenario = json.loads(json_str)
                else:
                    scenario = json.loads(response)  # Direct JSON parsing
                return scenario
            except Exception as e:
                logging.error(f"Failed to interpret paragraph: {e}")
                # Fallback scenario if interpretation fails
                return {
                    "title": "Generic Scenario",
                    "background": scenario_text[:100] + "..." if len(scenario_text) > 100 else scenario_text,
                    "initial_situation": "You’re facing a situation based on your input.",
                    "decision_points": [{
                        "id": "dp1",
                        "name": "Initial Decision",
                        "situation": "Decide how to proceed.",
                        "options": [
                            {"id": "A", "text": "Take initiative", "implications": "May resolve quickly"},
                            {"id": "B", "text": "Seek advice", "implications": "May delay but informed choice"}
                        ],
                        "best_practice": "A",
                        "best_practice_explanation": "Proactive steps often yield faster results."
                    }]
                }

    def start(self):
        if self.running:
            print("Scenario Voice AI is already running.")
            return
        self.running = True
        self.listening_thread = threading.Thread(target=self._listen_continuously)
        self.processing_thread = threading.Thread(target=self._process_audio_queue)
        self.listening_thread.start()
        self.processing_thread.start()
        introduction = f"Welcome to your scenario: {self.scenario['title']}. {self.scenario['background']} Let's start with: {self.scenario['initial_situation']} Would you like me to present the first decision point?"
        print("Starting scenario with introduction...")
        self._speak(introduction)

    def stop(self):
        self.running = False
        if self.listening_thread:
            self.listening_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        print("Scenario Voice AI stopped.")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        if indata is None or len(indata) == 0:
            return
        audio_data = indata.copy()
        volume_norm = np.linalg.norm(audio_data) / np.sqrt(len(audio_data))
        if self.recording and frames % 160 == 0:
            print(f"Volume level: {volume_norm:.4f}")
        if volume_norm > self.silence_threshold:
            if not self.recording and not self.speaking:
                print("Speech detected, starting recording...")
                self.recording = True
                self.speech_start_time = time.time()
                self.recording_buffer = [audio_data]
            elif self.recording:
                self.last_sound = time.time()
                self.recording_buffer.append(audio_data)
        elif self.recording:
            self.recording_buffer.append(audio_data)
            if time.time() - self.last_sound > self.silence_duration:
                speech_duration = time.time() - self.speech_start_time
                if speech_duration >= self.min_speech_duration:
                    print(f"Speech ended after {speech_duration:.2f} seconds, processing...")
                    audio_data = np.vstack(self.recording_buffer)
                    self.audio_queue.put(audio_data)
                else:
                    print(f"Speech too short ({speech_duration:.2f}s), ignoring...")
                self.recording = False
                self.recording_buffer = []

    def _listen_continuously(self):
        print("Listening for user input...")
        try:
            with sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                dtype="float32",
                blocksize=0
            ):
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error in listening thread: {e}")
            self.running = False

    def _process_audio_queue(self):
        while self.running:
            try:
                if not self.audio_queue.empty() and not self.speaking:
                    audio_data = self.audio_queue.get()
                    print("Transcribing speech...")
                    text = self._speech_to_text(audio_data)
                    if text and text.strip():
                        print(f"User said: {text}")
                        self.conversation_history.append({"role": "user", "content": text})
                        print("Generating response...")
                        response = self._generate_response(text)
                        print(f"AI response: {response}")
                        self.conversation_history.append({"role": "assistant", "content": response})
                        self._speak(response)
                    else:
                        print("No speech detected or couldn't transcribe")
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in processing thread: {e}")

    def _speech_to_text(self, audio_data):
        try:
            audio_array = audio_data.flatten().astype(np.float32)
            result = self.speech_recognition({"raw": audio_array, "sampling_rate": self.sample_rate})
            return result["text"]
        except Exception as e:
            print(f"Error in speech-to-text: {e}")
            return ""

    def _generate_response(self, user_input):
        try:
            user_input_lower = user_input.lower()
            if "scenario" in user_input_lower or "what's this about" in user_input_lower:
                return f"This is about: {self.scenario['title']}. {self.scenario['background']} The current situation is: {self.scenario['initial_situation']}"
            for point in self.scenario["decision_points"]:
                if point["id"].lower() in user_input_lower or point["name"].lower() in user_input_lower:
                    self.current_decision_point = point
                    options_text = "\n".join([f"Option {opt['id']}: {opt['text']}" for opt in point["options"]])
                    return f"Let's focus on {point['name']}. {point['situation']}\n\nYour options are:\n{options_text}\n\nWhich option would you choose?"
            if "options" in user_input_lower or "what can i do" in user_input_lower:
                if self.current_decision_point:
                    options_text = "\n".join([f"Option {opt['id']}: {opt['text']}" for opt in self.current_decision_point["options"]])
                    return f"For {self.current_decision_point['name']}, your options are:\n{options_text}\n\nWhich option would you choose?"
                else:
                    self.current_decision_point = self.scenario["decision_points"][0]
                    options_text = "\n".join([f"Option {opt['id']}: {opt['text']}" for opt in self.current_decision_point["options"]])
                    return f"Starting with {self.current_decision_point['name']}. {self.current_decision_point['situation']}\n\nYour options are:\n{options_text}\n\nWhich option would you choose?"
            if self.current_decision_point:
                for option in self.current_decision_point["options"]:
                    option_indicators = [
                        f"option {option['id']}".lower(),
                        f"choose {option['id']}".lower(),
                        f"pick {option['id']}".lower(),
                        f"select {option['id']}".lower(),
                        f"go with {option['id']}".lower(),
                        option['id'].lower()
                    ]
                    if any(indicator in user_input_lower for indicator in option_indicators):
                        best_practice = self.current_decision_point["best_practice"]
                        best_practice_explanation = self.current_decision_point["best_practice_explanation"]
                        if option["id"] == best_practice:
                            response = f"Great choice with Option {option['id']}! {option['implications']} This is indeed the best practice approach: {best_practice_explanation}"
                        else:
                            best_option = next((opt for opt in self.current_decision_point["options"] if opt["id"] == best_practice), None)
                            response = f"I understand your choice of Option {option['id']}. {option['implications']} However, Option {best_practice} ({best_option['text']}) is generally considered best practice because {best_practice_explanation}"
                        current_index = next((i for i, point in enumerate(self.scenario["decision_points"]) if point["id"] == self.current_decision_point["id"]), -1)
                        if current_index < len(self.scenario["decision_points"]) - 1:
                            next_point = self.scenario["decision_points"][current_index + 1]
                            self.current_decision_point = next_point
                            response += f"\n\nLet's move on to: {next_point['name']}. {next_point['situation']}\n\nYour options are:\n" + "\n".join([f"Option {opt['id']}: {opt['text']}" for opt in next_point["options"]])
                        else:
                            response += "\n\nYou've completed all decision points. What next?"
                            self.current_decision_point = None
                        return response
            scenario_context = f"""
            Title: {self.scenario.get('title', 'Custom Scenario')}
            Background: {self.scenario.get('background', 'User-provided context')}
            Initial Situation: {self.scenario.get('initial_situation', 'User-defined starting point')}
            Decision Points: {json.dumps(self.scenario.get('decision_points', []), indent=2)}
            """
            recent_history = self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
            conversation_text = "\n".join([f"{'User' if msg['role'] == 'user' else 'AI'}: {msg['content']}" for msg in recent_history])
            prompt = f"""
            You are an AI coach guiding a user through a custom workplace scenario.
            Scenario Context:
            {scenario_context}
            Recent conversation:
            {conversation_text}
            User said: "{user_input}"
            Respond conversationally, guiding them through the scenario or answering their query.
            If decision points exist and no choice is made, suggest options.
            """
            response = self.language_model.invoke(prompt).content
            return response.strip()
        except Exception as e:
            logging.error(f"Error in response generation: {e}")
            return "Sorry, I couldn’t process that. Could you try again?"

    def _speak(self, text):
        try:
            print("Speaking response...")
            self.speaking = True
            wav_file = "temp_speech.wav"
            self.tts.tts_to_file(text=text, file_path=wav_file, speaker="p339")
            audio = AudioSegment.from_wav(wav_file)
            play(audio)
            if os.path.exists(wav_file):
                os.remove(wav_file)
            self.speaking = False
            print("Done speaking, listening again...")
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            self.speaking = False

    def _speak_to_file(self, text, file_path):
        try:
            self.tts.tts_to_file(text=text, file_path=file_path, speaker="p339")
            return True
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            return False

def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    try:
        scenario_input = input("Enter your scenario (paragraph or JSON): ")
        agent = ScenarioVoiceAI(scenario_input=scenario_input, api_key=api_key)
        agent.start()
        print("Press Ctrl+C to stop the agent")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping the agent...")
    except ValueError as e:
        print(f"Error: {e}")
    finally:
        if 'agent' in locals() and agent:
            agent.stop()

if __name__ == "__main__":
    main()