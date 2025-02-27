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
from dotenv import load_dotenv  

load_dotenv()

class ScenarioVoiceAI:
    """
    A real-time voice AI agent that guides users through workplace scenarios,
    processing audio input and responding with voice guidance based on scenario context.
    """
    
    def __init__(self, scenario_data=None, api_key=None):
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
        
        # Load the scenario
        self.scenario = scenario_data if scenario_data else self._load_default_scenario()
        
        # Initialize models
        print("Initializing speech recognition model...")
        self.speech_recognition = pipeline("automatic-speech-recognition", model="openai/whisper-base")
        
        print("Initializing Google Gemini language model via Langchain...")
        self.language_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024,
        )
        
        print("Initializing text-to-speech model...")
        self.tts = TTS("tts_models/en/vctk/vits") 
        
        # Audio recording parameters
        self.sample_rate = 16000
        self.recording_buffer = []
        self.silence_threshold = 0.02
        self.silence_duration = 1.5  # seconds
        self.min_speech_duration = 1.0  # seconds
        self.last_sound = time.time()
        self.speech_start_time = None
        
        print("Scenario Voice AI initialized and ready!")
    
    def _load_default_scenario(self):
        """Load the default scenario about navigating a micromanaging manager."""
        return {
            "title": "Navigating a Micromanaging Manager and Project Setbacks",
            "learning_objectives": {
                "primary_skill": "Effectively escalating concerns to a skip-level manager while maintaining professionalism and focusing on business impact.",
                "secondary_skills": ["Active listening", "Strategic communication", "Problem-solving"],
                "key_takeaways": "How to present concerns about management that focuses on business impact, how to demonstrate confidence in your value, and how to seek constructive solutions."
            },
            "background": "You are a project manager leading a critical initiative for your company, but you've been struggling with your direct manager's excessive micromanagement, which has led to team frustration and project delays. Your manager consistently questions every decision, interferes with the team's workflow, and has created an atmosphere of distrust. You've attempted to address these concerns with your manager directly, but they have been dismissive, and the behavior has intensified. You realize that you need to escalate your concerns to your skip-level manager to protect the project, your team, and your ability to do your job. You are committed to the success of this project and have been recognized for your previous work.",
            "characters": {
                "protagonist": {
                    "name": "Sarah",
                    "role": "Project Manager",
                    "experience": "mid-career with 7+ years of experience",
                    "goals": "Successfully complete the project, protect her team's morale, improve the work environment, and advance in her career."
                },
                "stakeholders": [
                    {
                        "name": "David",
                        "role": "Direct Manager",
                        "style": "Micromanaging, insecure, resistant to feedback",
                        "motivations": "Maintaining control and avoiding blame"
                    },
                    {
                        "name": "Emily",
                        "role": "Skip-Level Manager",
                        "style": "Results-oriented, values efficiency and team morale, but might be unaware of day-to-day issues with direct reports",
                        "motivations": "Project success, maintaining a positive work environment, and overall company performance"
                    }
                ]
            },
            "initial_situation": "Sarah has just received feedback on a report from David that requires major changes to the work the team had completed in the past two days, which will push the project past the deadline and demoralize the team. David has also sent out an email questioning Sarah's decisions on the project. Sarah decides she needs to escalate this issue to Emily.",
            "decision_points": [
                {
                    "id": "approaching_emily",
                    "name": "Approaching Emily",
                    "situation": "Sarah needs to decide how to approach Emily for a meeting.",
                    "options": [
                        {
                            "id": "A",
                            "text": "Send a brief email requesting a meeting to discuss the project and her manager's leadership.",
                            "implications": "Allows for a professional and neutral beginning but might not convey the urgency."
                        },
                        {
                            "id": "B",
                            "text": "Send an email detailing all of the specific examples of David's behavior and request a meeting.",
                            "implications": "Provides context to the issues but might be overwhelming and perceived as complaining rather than problem-solving."
                        },
                        {
                            "id": "C",
                            "text": "Walk into Emily's office to express her concerns immediately.",
                            "implications": "Might convey urgency but is less professional, and it might put Emily on the defensive."
                        }
                    ],
                    "best_practice": "A",
                    "best_practice_explanation": "This approach is professional, gives a heads-up to Emily about the topic, and gives her time to consider the issues."
                },
                {
                    "id": "presenting_issues",
                    "name": "Presenting Issues",
                    "situation": "In the meeting with Emily, Sarah has to decide how to present the issues with David.",
                    "options": [
                        {
                            "id": "A",
                            "text": "Focus primarily on her personal frustrations and feelings of being micromanaged.",
                            "implications": "Could be perceived as unprofessional and not focused on the business."
                        },
                        {
                            "id": "B",
                            "text": "Provide specific examples of David's actions and explain how they have negatively affected the team, morale, and project timelines.",
                            "implications": "Demonstrates a focus on impact and the company and a desire to find solutions rather than complain."
                        },
                        {
                            "id": "C",
                            "text": "Briefly mention the issues and focus only on asking Emily to remove David from the project.",
                            "implications": "Could be viewed as an attempt to avoid conflict or as lacking professionalism by not presenting clear examples of the problems."
                        }
                    ],
                    "best_practice": "B",
                    "best_practice_explanation": "This approach uses a business focus, shows that Sarah is not simply complaining, and demonstrates a clear understanding of the issues."
                },
                {
                    "id": "response_and_recommendations",
                    "name": "Response and Recommendations",
                    "situation": "After explaining the situation, Sarah needs to decide how to respond to Emily's questions and how to present her recommendations for next steps.",
                    "options": [
                        {
                            "id": "A",
                            "text": "Agree to any suggestions and offer no recommendations of her own.",
                            "implications": "Shows a lack of confidence and could lead to further problems."
                        },
                        {
                            "id": "B",
                            "text": "Defend her decisions, focus on the negative and dismiss any attempts by Emily to understand the problem from David's point of view.",
                            "implications": "Could damage her credibility and prevent a positive solution."
                        },
                        {
                            "id": "C",
                            "text": "Listen carefully to Emily's response, ask clarifying questions, and present specific, actionable recommendations for moving forward.",
                            "implications": "Demonstrates professionalism, confidence, and a commitment to finding constructive solutions."
                        }
                    ],
                    "best_practice": "C",
                    "best_practice_explanation": "This approach demonstrates active listening, openness to feedback, and a commitment to finding a solution."
                }
            ],
            "key_lessons": [
                "Focus on Business Impact: Frame your concerns in terms of how the manager's actions impact the project goals, team morale, and overall business objectives.",
                "Unshakable Confidence: Express your concerns with confidence and clearly articulate how your skills are benefiting the company.",
                "Propose Solutions: Always approach with a solution-oriented mindset by offering potential solutions rather than simply stating problems.",
                "Active Listening: Pay close attention to how your audience responds and ask clarifying questions to understand their point of view."
            ],
            "critical_success_factors": [
                "Maintaining a professional demeanor, even when expressing negative feedback.",
                "Presenting concrete examples of the issues.",
                "Being assertive but not aggressive."
            ],
            "common_mistakes": [
                "Focusing solely on personal grievances.",
                "Using generalizations or accusatory language.",
                "Not being prepared to offer solutions.",
                "Becoming defensive or dismissive during the meeting."
            ]
        }
    
    def start(self):
        """Start the scenario voice AI agent."""
        if self.running:
            print("Scenario Voice AI is already running.")
            return
        
        self.running = True
        self.listening_thread = threading.Thread(target=self._listen_continuously)
        self.processing_thread = threading.Thread(target=self._process_audio_queue)
        
        self.listening_thread.start()
        self.processing_thread.start()
        
        # Initial greeting and scenario introduction
        introduction = f"Welcome to the scenario: {self.scenario['title']}. {self.scenario['background']} Let's start with the initial situation: {self.scenario['initial_situation']} Would you like me to present the first decision point?"
        print("Starting scenario with introduction...")
        self._speak(introduction)
    
    def stop(self):
        """Stop the scenario voice AI agent."""
        self.running = False
        if self.listening_thread:
            self.listening_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        print("Scenario Voice AI stopped.")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream to collect audio data."""
        if status:
            print(f"Audio status: {status}")

        if indata is None or len(indata) == 0:
            return  # Prevents processing empty data

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
        """Continuously listen for user input."""
        print("Listening for user input...")

        try:
            with sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                dtype="float32",
                blocksize=0  # Let system determine best size
            ):
                while self.running:
                    time.sleep(0.1)  # Prevent CPU overuse
        except Exception as e:
            print(f"Error in listening thread: {e}")
            self.running = False
    
    def _process_audio_queue(self):
        """Process audio data in the queue."""
        while self.running:
            try:
                if not self.audio_queue.empty() and not self.speaking:
                    audio_data = self.audio_queue.get()
                    
                    # Convert audio to text
                    print("Transcribing speech...")
                    text = self._speech_to_text(audio_data)
                    if text and text.strip():
                        print(f"User said: {text}")
                        
                        # Add to conversation history
                        self.conversation_history.append({"role": "user", "content": text})
                        
                        # Generate response
                        print("Generating response...")
                        response = self._generate_response(text)
                        print(f"AI response: {response}")
                        
                        # Add AI response to history
                        self.conversation_history.append({"role": "assistant", "content": response})
                        
                        # Convert response to speech
                        self._speak(response)
                    else:
                        print("No speech detected or couldn't transcribe")
                
                time.sleep(0.1)  # Small sleep to prevent CPU overuse
            except Exception as e:
                print(f"Error in processing thread: {e}")
    
    def _speech_to_text(self, audio_data):
        """Convert speech to text using the speech recognition model."""
        try:
            # Convert numpy array to the format expected by the model
            audio_array = audio_data.flatten().astype(np.float32)
            
            # Get transcription
            result = self.speech_recognition({"raw": audio_array, "sampling_rate": self.sample_rate})
            return result["text"]
        except Exception as e:
            print(f"Error in speech-to-text: {e}")
            return ""
    
    def _generate_response(self, user_input):
        """Generate a response based on the user input and the current scenario context."""
        try:
            user_input_lower = user_input.lower()
            
            # Check if the user is asking about the scenario or for help
            if any(keyword in user_input_lower for keyword in ["scenario", "what's this about", "tell me more", "explain the scenario"]):
                return f"We're working through the scenario: {self.scenario['title']}. {self.scenario['background']} The current situation is: {self.scenario['initial_situation']}"
            
            # Check if the user wants to move to or learn about a specific decision point
            for point in self.scenario["decision_points"]:
                if any(keyword in user_input_lower for keyword in [point["id"].lower(), point["name"].lower()]):
                    self.current_decision_point = point
                    options_text = "\n".join([f"Option {opt['id']}: {opt['text']}" for opt in point["options"]])
                    return f"Let's focus on the {point['name']} decision. {point['situation']}\n\nYour options are:\n{options_text}\n\nWhich option would you choose? Or would you like me to explain any option in more detail?"
            
            # Check if the user is asking about the current options
            if any(keyword in user_input_lower for keyword in ["options", "choices", "what can i do", "what are my options"]):
                if self.current_decision_point:
                    options_text = "\n".join([f"Option {opt['id']}: {opt['text']}" for opt in self.current_decision_point["options"]])
                    return f"For the {self.current_decision_point['name']} decision, your options are:\n{options_text}\n\nWhich option would you choose?"
                else:
                    # Start with the first decision point if none is active
                    self.current_decision_point = self.scenario["decision_points"][0]
                    options_text = "\n".join([f"Option {opt['id']}: {opt['text']}" for opt in self.current_decision_point["options"]])
                    return f"Let's start with the first decision: {self.current_decision_point['name']}. {self.current_decision_point['situation']}\n\nYour options are:\n{options_text}\n\nWhich option would you choose?"
            
            # Check if the user is making a choice
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
                            response = f"Great choice with Option {option['id']}! {option['implications']} This is indeed the best practice approach. {best_practice_explanation}"
                        else:
                            # Find the best practice option
                            best_option = next((opt for opt in self.current_decision_point["options"] if opt["id"] == best_practice), None)
                            response = f"I understand your choice of Option {option['id']}. {option['implications']} However, Option {best_practice} ({best_option['text']}) is generally considered best practice because {best_practice_explanation}"
                        
                        # Move to the next decision point if available
                        current_index = next((i for i, point in enumerate(self.scenario["decision_points"]) if point["id"] == self.current_decision_point["id"]), -1)
                        if current_index < len(self.scenario["decision_points"]) - 1:
                            next_point = self.scenario["decision_points"][current_index + 1]
                            self.current_decision_point = next_point
                            response += f"\n\nLet's move on to the next decision: {next_point['name']}. {next_point['situation']}\n\nYour options are:\n" + "\n".join([f"Option {opt['id']}: {opt['text']}" for opt in next_point["options"]])
                        else:
                            # We've completed all decision points, provide summary
                            key_lessons = "\n".join([f"• {lesson}" for lesson in self.scenario["key_lessons"]])
                            response += f"\n\nYou've completed all the decision points in this scenario. Here are the key lessons:\n{key_lessons}\n\nWould you like to discuss any particular aspect of the scenario in more detail?"
                            self.current_decision_point = None
                        
                        return response
            
            # For more general questions or statements, use the language model with scenario context
            # Prepare the context for the model
            scenario_context = f"""
            Title: {self.scenario['title']}
            
            Background: {self.scenario['background']}
            
            Initial Situation: {self.scenario['initial_situation']}
            
            Character Profiles:
            - Protagonist: {self.scenario['characters']['protagonist']['name']}, {self.scenario['characters']['protagonist']['role']}
            - Stakeholders:
              - {self.scenario['characters']['stakeholders'][0]['name']}, {self.scenario['characters']['stakeholders'][0]['role']}, {self.scenario['characters']['stakeholders'][0]['style']}
              - {self.scenario['characters']['stakeholders'][1]['name']}, {self.scenario['characters']['stakeholders'][1]['role']}, {self.scenario['characters']['stakeholders'][1]['style']}
            """
            
            # Add decision points if we're at a specific one
            if self.current_decision_point:
                scenario_context += f"\nCurrent Decision Point: {self.current_decision_point['name']}\n"
                scenario_context += f"Situation: {self.current_decision_point['situation']}\n"
                for option in self.current_decision_point["options"]:
                    scenario_context += f"Option {option['id']}: {option['text']} - {option['implications']}\n"
            
            # Format recent conversation history
            recent_history = self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
            conversation_text = ""
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "AI Coach"
                conversation_text += f"{role}: {msg['content']}\n\n"
            
            # Create the prompt
            prompt = f"""You are an AI coach helping a user work through a workplace scenario about dealing with a micromanaging manager.
            
            Scenario Context:
            {scenario_context}
            
            Recent conversation:
            {conversation_text}
            
            The user just said: "{user_input}"
            
            Respond as a helpful coach guiding them through the scenario. Be conversational but focused on helping them understand the best practices for handling workplace situations. 
            If they haven't made a clear decision and a decision point is active, guide them toward making a choice.
            If they're asking about a specific aspect of workplace communication or management, provide practical advice.
            Keep your response concise, empathetic, and actionable.
            """
            
            # Generate response from the language model
            response = self.language_model.invoke(prompt).content
            return response.strip()
            
        except Exception as e:
            print(f"Error in response generation: {e}")
            return "I'm sorry, I couldn't generate a proper response. Could you please try again or ask another question about the scenario?"
    
    def _speak(self, text):
        """Convert text to speech and play it."""
        try:
            print("Speaking response...")
            self.speaking = True
            
            # Generate speech
            wav_file = "temp_speech.wav"
            self.tts.tts_to_file(text=text, file_path=wav_file, speaker="p339")
            
            # Play the audio
            audio = AudioSegment.from_wav(wav_file)
            play(audio)
            
            # Clean up
            if os.path.exists(wav_file):
                os.remove(wav_file)
            
            self.speaking = False
            print("Done speaking, listening again...")
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            self.speaking = False

def main():
    """Main function to run the Scenario Voice AI."""
    # Check for API key in environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    try:
        # Initialize the agent with the micromanaging manager scenario
        agent = ScenarioVoiceAI(api_key=api_key)
        
        # Start the agent
        agent.start()
        
        # Keep the main thread running
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