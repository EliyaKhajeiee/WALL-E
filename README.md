# WALL-E Voice Assistant

A voice-activated AI assistant powered by Ollama, with Wikipedia knowledge and conversation memory.

## Features

- 🎤 **Wake word detection**: Say "Hi WALL-E" to activate
- 🗣️ **Speech-to-text**: Natural voice commands via microphone
- 🔊 **Text-to-speech**: Responds through speakers
- 📚 **Wikipedia knowledge**: 31,000+ articles vectorized
- 🧠 **Conversation memory**: Remembers and summarizes conversations
- ⚡ **Lazy loading**: Instant startup with persistent vector database
- 🤖 **Ollama-powered**: Runs locally with llama3

## Quick Start (Raspberry Pi)

### 1. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```

### 2. Clone and Setup
```bash
git clone <your-repo-url> ~/wall-e
cd ~/wall-e

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip portaudio19-dev espeak
pip3 install -r requirements_voice.txt
```

### 3. Load Wikipedia Data (First Time Only)
```bash
# This will take time but only needs to be done once
python3 setup_initial_db.py
```

### 4. Run WALL-E
```bash
python3 walle_voice.py
```

Say **"Hi WALL-E"** and start talking!

## Files

- `walle_voice.py` - Main voice assistant script
- `walle_rag.py` - Text-based RAG system (for testing)
- `walle_data/learned_data.json` - Personal facts and memories
- `walle_data/wiki_*.json` - Wikipedia articles (31K+ files)
- `requirements_voice.txt` - Python dependencies
- `RASPBERRY_PI_SETUP.md` - Detailed setup guide

## Usage

1. Say: **"Hi WALL-E"** (wake word)
2. Wait for beep/response
3. Ask your question
4. WALL-E responds via speakers

## Configuration

Edit `walle_voice.py`:
- `MODEL_NAME`: Change Ollama model
- `WAKE_WORD`: Change wake phrase
- TTS speed/volume: Adjust `tts_engine.setProperty()`

## Autostart on Boot

See `RASPBERRY_PI_SETUP.md` for systemd service configuration.

## Requirements

- Raspberry Pi 3B+ or newer
- Microphone (USB or GPIO)
- Speakers (3.5mm, USB, or Bluetooth)
- 16GB+ SD card (for Wikipedia data)
- Internet connection (initial setup only)

## License

MIT
