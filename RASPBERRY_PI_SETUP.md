# WALL-E Voice Assistant - Raspberry Pi Setup

## Prerequisites
- Raspberry Pi (3B+ or newer recommended)
- Microphone (USB or via GPIO)
- Speakers (via 3.5mm jack, USB, or Bluetooth)
- Internet connection

## Installation Steps

### 1. System Setup
```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y python3-pip python3-dev portaudio19-dev espeak ffmpeg

# Install ALSA tools for audio
sudo apt-get install -y alsa-utils

# Test microphone
arecord -l

# Test speakers
speaker-test -t wav -c 2
```

### 2. Install Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the llama3 model
ollama pull llama3

# Test it
ollama run llama3 "Say hello"
```

### 3. Install Python Dependencies
```bash
# Create virtual environment
cd ~/wall-e
python3 -m venv wall-e-env
source wall-e-env/bin/activate

# Install core dependencies
pip install langchain-community langchain-huggingface langchain-chroma

# Install voice dependencies
pip install SpeechRecognition pyttsx3 pyaudio

# For better TTS on Pi, optionally use espeak
pip install python-espeak
```

### 4. Transfer Your Files
```bash
# On your PC, copy files to Pi
scp -r walle_data/ pi@<raspberry-pi-ip>:~/wall-e/
scp -r vector_db/ pi@<raspberry-pi-ip>:~/wall-e/
scp sync_state.json pi@<raspberry-pi-ip>:~/wall-e/
scp walle_voice.py pi@<raspberry-pi-ip>:~/wall-e/
```

### 5. Configure Audio

**Test Microphone:**
```bash
# Record 5 seconds
arecord -d 5 test.wav

# Play it back
aplay test.wav
```

**Set Default Audio Device:**
```bash
# List audio devices
aplay -l
arecord -l

# Edit ALSA config if needed
nano ~/.asoundrc
```

Example `.asoundrc`:
```
pcm.!default {
    type hw
    card 1
}

ctl.!default {
    type hw
    card 1
}
```

### 6. Run WALL-E Voice Assistant
```bash
cd ~/wall-e
source wall-e-env/bin/activate
python3 walle_voice.py
```

## Usage

1. Say: **"Hi WALL-E"** (wake word)
2. Wait for WALL-E to beep
3. Ask your question
4. WALL-E will respond via speakers

## Autostart on Boot (Optional)

Create a systemd service:

```bash
sudo nano /etc/systemd/system/walle.service
```

Add this content:
```ini
[Unit]
Description=WALL-E Voice Assistant
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/wall-e
Environment="PATH=/home/pi/wall-e/wall-e-env/bin"
ExecStart=/home/pi/wall-e/wall-e-env/bin/python3 /home/pi/wall-e/walle_voice.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable walle.service
sudo systemctl start walle.service

# Check status
sudo systemctl status walle.service

# View logs
journalctl -u walle.service -f
```

## Troubleshooting

### No audio output
```bash
# Check ALSA mixer
alsamixer

# Unmute and increase volume
```

### Microphone not working
```bash
# Check permissions
sudo usermod -a -G audio pi

# Reboot
sudo reboot
```

### Wake word not detecting
- Speak clearly and closer to the microphone
- Reduce background noise
- Try adjusting `recognizer.energy_threshold` in the code

### Ollama connection issues
```bash
# Check if Ollama is running
systemctl status ollama

# Restart if needed
sudo systemctl restart ollama
```

## Performance Tips

1. **Reduce model size**: Use `ollama pull llama3:8b` for faster responses
2. **Adjust TTS speed**: Change `tts_engine.setProperty('rate', 150)` to your preference
3. **Limit context**: The script already retrieves only top 5 relevant docs
4. **Use lighter embeddings**: Already using MiniLM (fast and lightweight)

## Notes

- First query after boot takes ~10-30 seconds (model loading)
- Subsequent queries are much faster
- The system uses lazy loading - wiki files stay in vector DB, no reload needed
- Conversations are auto-summarized every 8 turns to maintain memory
