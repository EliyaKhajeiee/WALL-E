#!/bin/bash
# Install Whisper and Piper for offline voice assistant

echo "Installing offline voice components..."

# Install Whisper.cpp for fast local transcription
echo "Installing Whisper.cpp..."
cd ~
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make

# Download tiny model (fast, good for wake words)
bash ./models/download-ggml-model.sh tiny

# Add to PATH
echo 'export PATH=$PATH:~/whisper.cpp' >> ~/.bashrc

# Install Piper TTS (optional, we're using espeak as fallback)
echo "Piper setup skipped - using espeak for now"

echo ""
echo "Installation complete!"
echo "Run: source ~/.bashrc"
echo "Then: python3 ~/wall-e/walle_voice_offline.py"
