# walle_voice.py
# -*- coding: utf-8 -*-
"""
WALL-E Voice Assistant for Raspberry Pi
Listens for "Hi WALL-E" wake word, then responds to questions
"""
import sys
import json
import threading
import queue
from pathlib import Path
from datetime import datetime

# Speech libraries
import subprocess
import pyaudio
import wave
import numpy as np

# Try gpiozero first (newer, better), fallback to RPi.GPIO
try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
    GPIO_METHOD = 'gpiozero'
except ImportError:
    try:
        import RPi.GPIO as GPIO
        GPIO_AVAILABLE = True
        GPIO_METHOD = 'RPi.GPIO'
    except ImportError:
        GPIO_AVAILABLE = False
        GPIO_METHOD = None
        print("[WARNING] No GPIO library available - button mode disabled")

# RAG components
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Fix encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# CONFIG
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "walle_data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
SYNC_FILE = BASE_DIR / "sync_state.json"
MODEL_NAME = "llama3.2:1b"  # Smaller model for Raspberry Pi
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# GPIO Button Configuration for Raspberry Pi HAT
# Common HAT buttons are on GPIO 17, 22, 23, 24, or 27
BUTTON_PIN = 17  # GPIO pin for the button (BCM numbering)

# Initialize components
print("[INIT] Starting WALL-E voice assistant...")
llm = Ollama(model=MODEL_NAME)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectordb = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)

# Speech components
try:
    import whisper
    whisper_model = whisper.load_model("tiny")  # Tiny model = ~75MB, fast
    print("[OK] Whisper loaded (offline mode)")
except ImportError:
    print("[ERROR] Please install: pip install openai-whisper")
    sys.exit(1)

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
audio_interface = pyaudio.PyAudio()

# Setup GPIO button if available
button = None
if GPIO_AVAILABLE:
    try:
        if GPIO_METHOD == 'gpiozero':
            button = Button(BUTTON_PIN, pull_up=True, bounce_time=0.1)
            print(f"[OK] Button on GPIO pin {BUTTON_PIN} ready! (using gpiozero)")
        else:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            print(f"[OK] Button on GPIO pin {BUTTON_PIN} ready! (using RPi.GPIO)")
    except Exception as e:
        print(f"[WARNING] GPIO setup failed: {e}")
        print("[INFO] Falling back to always-listening mode (no button)")
        GPIO_AVAILABLE = False

# Conversation history
conversation_history = []
SUMMARIZE_AFTER_TURNS = 8

print("[OK] WALL-E voice assistant ready!\n")


def speak(text):
    """Convert text to speech - tries Piper first, falls back to espeak."""
    print(f"WALL-E: {text}")
    try:
        # Try Piper first (better quality, still offline)
        result = subprocess.run(
            ['piper', '--model', 'en_US-lessac-medium', '--output-raw'],
            input=text.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            # Play through aplay
            subprocess.run(['aplay', '-r', '22050', '-f', 'S16_LE', '-c', '1'],
                         input=result.stdout,
                         stderr=subprocess.DEVNULL)
        else:
            raise Exception("Piper failed")
    except:
        # Fallback to espeak (always works, offline)
        subprocess.run(['espeak', '-s', '150', '-a', '200', text],
                      stderr=subprocess.DEVNULL,
                      stdout=subprocess.DEVNULL)


def record_audio(duration=3):
    """Record audio and return as numpy array."""
    stream = audio_interface.open(format=FORMAT, channels=CHANNELS,
                                   rate=RATE, input=True,
                                   frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    # Convert to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize

    return audio_data


def wait_for_button_press():
    """Wait for button press on GPIO pin."""
    if not GPIO_AVAILABLE:
        print("[ERROR] GPIO not available!")
        return False

    print("[BUTTON] Press button to toggle...")

    try:
        if GPIO_METHOD == 'gpiozero':
            # gpiozero method
            button.wait_for_press()
            print("[BUTTON] Button pressed!")
            return True
        else:
            # RPi.GPIO method
            GPIO.wait_for_edge(BUTTON_PIN, GPIO.FALLING)
            print("[BUTTON] Button pressed!")
            return True
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[ERROR] Button error: {e}")
        return False


def listen_for_wake_word():
    """Listen for wake word with better noise filtering."""
    try:
        # Record short audio snippet
        audio_data = record_audio(duration=2)

        # Calculate volume to filter out silence
        volume = np.abs(audio_data).mean()
        if volume < 0.01:  # Too quiet, skip
            return False

        # Transcribe
        result = whisper_model.transcribe(audio_data, language='en',
                                         fp16=False, task='transcribe')
        text = result['text'].strip().lower()

        if text:
            print(f"[HEARD] '{text}'")

        # Check for wake words - more flexible matching
        wake_words = ['wall-e', 'walle', 'wally', 'walley', 'hi wall']
        for wake_word in wake_words:
            if wake_word in text:
                print(f"[WAKE] Wake word detected!")
                speak("Yes?")
                return True

        return False

    except Exception as e:
        print(f"[ERROR] Listen error: {e}")
        return False


def listen_for_command():
    """Listen for user command after button press."""
    print("[LISTENING] Listening for your question...")
    speak("Beep boop!")

    try:
        # Record 5 seconds for question
        audio_data = record_audio(duration=5)

        # Transcribe
        result = whisper_model.transcribe(audio_data, language='en',
                                         fp16=False, task='transcribe')
        text = result['text'].strip()

        if text:
            print(f"You: {text}")
            return text
        else:
            speak("I didn't hear anything. Beep boop.")
            return None

    except Exception as e:
        print(f"[ERROR] {e}")
        speak("Sorry, I had trouble hearing you.")
        return None


def build_context(user_input: str):
    """Retrieve relevant context from vector database."""
    try:
        # Reduce from k=5 to k=2 for faster inference
        retrieved = vectordb.similarity_search(user_input, k=2)
        context = "\n".join(
            [f"- {r.page_content[:200]}" for r in retrieved]  # Limit context length
        )
        return context
    except Exception as e:
        print(f"[WARNING] Error retrieving context: {e}")
        return ""


def query_walle(user_input: str):
    """Process query and generate response."""
    if not user_input:
        return

    print("[THINKING] Processing your question...")

    context = build_context(user_input)
    # Reduce conversation history from 5 to 2 for faster inference
    conversation = "\n".join(
        [f"You: {m['user']}\nWALL-E: {m['ai']}" for m in conversation_history[-2:]]
    )

    # Shorter, simpler prompt for faster inference
    prompt = f"""You are WALL-E. Answer in 1 SHORT sentence.

Facts: {context}

User: {user_input}
WALL-E:"""

    try:
        # Add num_predict to limit response length (faster)
        response = llm.invoke(prompt, num_predict=100).strip()

        # Clean up response if too long
        sentences = response.split('.')
        if len(sentences) > 2:
            response = '. '.join(sentences[:2]) + '.'

        conversation_history.append({"user": user_input, "ai": response})

        # Disable auto-summarization for speed
        # if len(conversation_history) >= SUMMARIZE_AFTER_TURNS:
        #     summarize_conversation()

        return response

    except Exception as e:
        print(f"[ERROR] {e}")
        return "Beep boop! Sorry, I'm having trouble thinking right now."


def summarize_conversation():
    """Summarize conversation and store in vector DB."""
    global conversation_history

    if len(conversation_history) < 4:
        return

    try:
        recent = conversation_history[-SUMMARIZE_AFTER_TURNS:]
        full = "\n".join([f"You: {m['user']}\nWALL-E: {m['ai']}" for m in recent])

        prompt = (
            "Summarize the following conversation into concise factual bullet points "
            "(1-2 sentences per fact):\n\n" + full + "\n\nSummary:"
        )

        summary = llm.invoke(prompt).strip()

        if summary:
            timestamp = datetime.utcnow().isoformat()
            vectordb.add_texts(
                [f"conversation_summary.{timestamp}: {summary}"],
                metadatas=[{"source": "conversation_summary", "ts": timestamp}],
            )
            print(f"[MEMORY] Stored conversation summary")

    except Exception as e:
        print(f"[WARNING] Error summarizing: {e}")


def main():
    """Main voice assistant loop with button press-to-talk."""
    speak("Beep boop! Hi, I'm WALL-E!")

    if GPIO_AVAILABLE:
        speak("Press the button to talk to me!")
        print("\n[READY] Press button to ask a question - no wake word needed!")

        while True:
            try:
                # Wait for button press
                if wait_for_button_press():
                    # Get command immediately - no wake word needed!
                    command = listen_for_command()

                    if command:
                        # Process and respond
                        response = query_walle(command)
                        speak(response)

                    print("\n[READY] Press button to ask another question!")

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Goodbye!")
                speak("Beep boop! Goodbye!")
                if GPIO_AVAILABLE and GPIO_METHOD == 'RPi.GPIO':
                    GPIO.cleanup()
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

    else:
        # Fallback to always-listening mode if no GPIO
        speak("Say 'Hi WALL-E' to talk to me!")

        while True:
            try:
                # Wait for wake word
                if listen_for_wake_word():
                    # Get command
                    command = listen_for_command()

                    if command:
                        # Process and respond
                        response = query_walle(command)
                        speak(response)

                    print("\n[LISTENING] Waiting for wake word...\n")

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Goodbye!")
                speak("Beep boop! Goodbye!")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                continue


if __name__ == "__main__":
    main()
