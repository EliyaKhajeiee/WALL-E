# walle_voice_offline.py
# -*- coding: utf-8 -*-
"""
WALL-E Voice Assistant - Fully Offline Version
Uses Whisper for speech recognition and Piper for TTS
No internet required!
"""
import sys
import json
import wave
import subprocess
import pyaudio
from pathlib import Path
from datetime import datetime

# RAG components
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# CONFIG
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "walle_data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
MODEL_NAME = "llama3"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WAKE_WORD = "hi walle"

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3

print("[INIT] Starting WALL-E offline voice assistant...")
llm = Ollama(model=MODEL_NAME)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectordb = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)

# Audio
audio = pyaudio.PyAudio()

# Conversation history
conversation_history = []
SUMMARIZE_AFTER_TURNS = 8

print("[OK] WALL-E offline voice assistant ready!\n")


def speak(text):
    """Convert text to speech using Piper."""
    print(f"WALL-E: {text}")
    try:
        # Use piper for TTS (need to install: pip install piper-tts)
        # Or use espeak as fallback
        subprocess.run(['espeak', '-s', '150', '-a', '200', text],
                      stderr=subprocess.DEVNULL,
                      stdout=subprocess.DEVNULL)
    except Exception as e:
        print(f"[WARNING] Speech output failed: {e}")


def record_audio(duration=3):
    """Record audio from microphone."""
    stream = audio.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)

    print(f"[RECORDING] Recording for {duration} seconds...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    # Save to WAV file
    temp_file = BASE_DIR / "temp_audio.wav"
    wf = wave.open(str(temp_file), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return temp_file


def transcribe_with_whisper(audio_file):
    """Transcribe audio using Whisper."""
    try:
        # Use whisper.cpp for fast local transcription
        result = subprocess.run(
            ['whisper', '--model', 'tiny', '--language', 'en',
             '--output-format', 'txt', str(audio_file)],
            capture_output=True,
            text=True
        )

        # Read the output file
        txt_file = audio_file.with_suffix('.txt')
        if txt_file.exists():
            text = txt_file.read_text().strip()
            txt_file.unlink()  # Clean up
            return text.lower()

        return None
    except Exception as e:
        print(f"[ERROR] Whisper transcription failed: {e}")
        return None


def listen_for_wake_word():
    """Listen for 'Hi WALL-E' wake word."""
    print("[LISTENING] Waiting for wake word 'Hi WALL-E'...")

    while True:
        try:
            # Record audio
            audio_file = record_audio(duration=2)

            # Transcribe
            text = transcribe_with_whisper(audio_file)

            if text:
                print(f"[HEARD] '{text}'")

                if WAKE_WORD in text:
                    print(f"[WAKE] Detected!")
                    audio_file.unlink()  # Clean up
                    return True

            audio_file.unlink()  # Clean up

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[ERROR] {e}")
            continue


def listen_for_command():
    """Listen for user command after wake word."""
    print("[LISTENING] Listening for your question...")
    speak("Beep boop! How can I help?")

    try:
        # Record longer audio for question
        audio_file = record_audio(duration=5)

        # Transcribe
        text = transcribe_with_whisper(audio_file)

        audio_file.unlink()  # Clean up

        if text:
            print(f"You: {text}")
            return text
        else:
            speak("Sorry, I didn't hear anything.")
            return None

    except Exception as e:
        print(f"[ERROR] {e}")
        speak("Sorry, I had trouble hearing you.")
        return None


def build_context(user_input: str):
    """Retrieve relevant context from vector database."""
    try:
        retrieved = vectordb.similarity_search(user_input, k=5)
        context = "\n".join([f"- {r.page_content}" for r in retrieved])
        return context
    except Exception as e:
        print(f"[WARNING] Error retrieving context: {e}")
        return ""


def query_walle(user_input: str):
    """Process query and generate response."""
    if not user_input:
        return

    context = build_context(user_input)
    conversation = "\n".join(
        [f"You: {m['user']}\nWALL-E: {m['ai']}" for m in conversation_history[-5:]]
    )

    prompt = f"""You are WALL-E, a friendly robot companion. Answer in 1-2 SHORT sentences maximum.
Be helpful, warm, and use simple robot sounds like "beep boop" occasionally. Keep it BRIEF.

Facts:
{context}

Recent chat:
{conversation}

User: {user_input}
WALL-E:"""

    try:
        response = llm.invoke(prompt).strip()

        # Clean up response if too long
        sentences = response.split('.')
        if len(sentences) > 2:
            response = '. '.join(sentences[:2]) + '.'

        conversation_history.append({"user": user_input, "ai": response})

        # Periodically summarize
        if len(conversation_history) >= SUMMARIZE_AFTER_TURNS:
            summarize_conversation()

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
    """Main voice assistant loop."""
    speak("Beep boop! Hi, I'm WALL-E! Say 'Hi WALL-E' to talk to me!")

    try:
        while True:
            # Wait for wake word
            if listen_for_wake_word():
                # Get command
                command = listen_for_command()

                if command:
                    # Process and respond
                    response = query_walle(command)
                    speak(response)

                print("\n[LISTENING] Waiting for wake word 'Hi WALL-E'...\n")

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Goodbye!")
        speak("Beep boop! Goodbye!")
        audio.terminate()
    except Exception as e:
        print(f"[ERROR] {e}")
        audio.terminate()


if __name__ == "__main__":
    main()
