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
import speech_recognition as sr
import subprocess

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
MODEL_NAME = "llama3"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WAKE_WORD = "hi walle"

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
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Conversation history
conversation_history = []
SUMMARIZE_AFTER_TURNS = 8

print("[OK] WALL-E voice assistant ready!\n")


def speak(text):
    """Convert text to speech using espeak directly."""
    print(f"WALL-E: {text}")
    try:
        # Use espeak directly (simpler and more reliable on Pi)
        subprocess.run(['espeak', '-s', '150', '-a', '200', text],
                      stderr=subprocess.DEVNULL,
                      stdout=subprocess.DEVNULL)
    except Exception as e:
        print(f"[WARNING] Speech output failed: {e}")


def listen_for_wake_word():
    """Listen for 'Hi WALL-E' wake word."""
    with microphone as source:
        print("[LISTENING] Waiting for wake word 'Hi WALL-E'...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        while True:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                text = recognizer.recognize_google(audio).lower()

                if WAKE_WORD in text:
                    print(f"[WAKE] Detected: '{text}'")
                    return True

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print(f"[ERROR] {e}")
                continue


def listen_for_command():
    """Listen for user command after wake word."""
    with microphone as source:
        print("[LISTENING] Listening for your question...")
        speak("Beep boop! How can I help?")

        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Beep boop.")
            return None
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand that.")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")
            speak("Sorry, I had trouble hearing you.")
            return None


def build_context(user_input: str):
    """Retrieve relevant context from vector database."""
    try:
        retrieved = vectordb.similarity_search(user_input, k=5)
        context = "\n".join(
            [f"- {r.page_content}" for r in retrieved]
        )
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

        # Summarize periodically
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

                print("\n[LISTENING] Waiting for wake word 'Hi WALL-E'...\n")

        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Goodbye!")
            speak("Beep boop! Goodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            continue


if __name__ == "__main__":
    main()
