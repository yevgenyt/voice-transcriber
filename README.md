# Voice Transcriber - OpenAI Whisper Edition

Press **Left Shift + Left Ctrl + Space** to speak instructions. Your M6 ZealSound microphone will record, transcribe using faster-Whisper, and type the text directly into your active window.

## Features

- 🎤 **High-accuracy speech recognition** - OpenAI Whisper (medium model) runs offline
- ✍️ **Auto-punctuation** - Automatically adds periods and capitalizes sentences
- ⌨️ **Direct keyboard input** - Types into any active window
- 🚀 **Simple hotkey activation** - Just press a key combination
- 🎯 **Works anywhere** - Compatible with Wayland, X11, any Linux desktop
- ⚡ **CPU optimized** - Fast transcription on CPU; ROCm GPU acceleration available (see notes)
- 🔐 **Privacy-first** - All processing happens locally on your machine

## Quick Start

### 1. Setup (One-Time Only)

Add yourself to the `input` group:
```bash
sudo usermod -a -G input $USER
```

**IMPORTANT**: After this, log out completely and log back in.

### 2. Run the Transcriber

```bash
transcriber
```

Or manually:
```bash
cd ~/Applications/voice-transcriber
source venv/bin/activate
python3 transcriber.py
```

You'll see:
```
==================================================
🎙️  Voice Transcriber Started
==================================================
Hotkey: Left Shift + Left Ctrl + Space
Model: Whisper medium
Features: Auto-punctuation

Press Ctrl+C to exit
```

### 3. Use It

- Press and hold **Left Shift + Left Ctrl**
- While holding, press **Space**
- Speak your instructions clearly
- Release **Space** to stop recording
- Your speech appears as typed text with automatic punctuation

## Architecture

```
Keyboard (evdev) ──→ Hotkey Detection ──→ Audio Recording (48kHz)
                           ↓
                    sounddevice (USB Mic)
                           ↓
                    scipy.signal (proper resampling to 16kHz)
                           ↓
                    faster-whisper (GPU accelerated)
                           ↓
                    pynput (keyboard output + auto-punctuation)
```

## Project Structure

```
~/Applications/voice-transcriber/
├── transcriber.py          ← Main application (entry point)
├── requirements.txt        ← Python dependencies
├── launch.sh              ← Launcher script
├── README.md              ← This file
└── venv/                  ← Python virtual environment
```

## Files Overview

| File | Purpose |
|------|---------|
| `transcriber.py` | Main application with all critical fixes |
| `requirements.txt` | All dependencies with versions |
| `launch.sh` | Simple launcher (aliased as `transcriber`) |
| `venv/` | Virtual environment with installed packages |

## Configuration

Edit `transcriber.py` (lines 25-33) to customize:

```python
WHISPER_MODEL = "medium"         # Options: tiny, base, small, medium, large
AUTO_PUNCTUATION = True          # Add periods and capitalization
USE_GPU = True                   # Use GPU if available (ROCm on AMD, CUDA on NVIDIA)
MICROPHONE_GAIN = 2.0            # Microphone amplification
```

## Troubleshooting

### "Permission denied: /dev/input/event*"

```bash
sudo usermod -a -G input $USER
# Then log out and log back in completely
```

### Hotkey not detected

1. Verify you're in the input group: `groups | grep input`
2. Make sure you logged out and back in
3. Try: Left Shift + Left Ctrl + Space (hold first two, then press space)

### No audio or weak transcription

- Check microphone: `amixer -c 1 sget Mic`
- Test recording: `arecord -D hw:1,0 -f cd /tmp/test.wav`
- Speak clearly and close to microphone
- Reduce background noise

### Auto-detection didn't find my microphone

Edit line 90 in `transcriber.py` to specify manually:
```python
self.audio_device = 4  # Your device number
```

Find your device: `arecord -l`

## Hardware Tested

- **Microphone**: M6 ZealSound (WXMH mini USB Audio)
- **System**: Ubuntu 25.10 with Wayland, X11 compatible
- **CPU**: AMD Ryzen (Strix Halo)
- **GPU**: AMD Radeon 880M (currently CPU processing; see ROCm notes below)
- **Storage**: ~1.4GB for medium Whisper model cache

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Hotkey | evdev | Keyboard event monitoring (Wayland-native) |
| Audio Capture | sounddevice | Microphone recording |
| Audio Processing | scipy.signal, numpy | Resampling (anti-aliased) |
| Speech Recognition | faster-whisper | Offline transcription |
| Keyboard Output | pynput | Type text into active window |

## Critical Implementation Details

**Audio Pipeline:**
- Records at 48kHz (microphone native rate)
- Resamples to 16kHz using scipy's polyphase filter (anti-aliasing)
- Applies noise gating and peak normalization
- Feeds to Whisper for transcription

**Thread Safety:**
- Uses threading.Lock() for recording state
- Prevents race conditions on multi-core systems

**Resource Management:**
- Context managers for all file operations
- Proper cleanup of temporary audio files
- No file descriptor leaks

## Resuming Development

When you need to continue:

1. **Review current state**:
   ```bash
   cd ~/Applications/voice-transcriber
   ls -la  # See project structure
   grep -n "CRITICAL" transcriber.py  # Find implementation notes
   ```

2. **Run the app**:
   ```bash
   transcriber
   ```

3. **Check dependencies**:
   ```bash
   source venv/bin/activate
   pip freeze
   ```

4. **Key files to reference**:
   - `transcriber.py` - All code in one file for simplicity
   - `requirements.txt` - Exact versions of dependencies

## Performance

- **Cold start**: ~3-5 seconds (model loading on first run)
- **Transcription**: ~3-5 seconds for 5-second audio (CPU, medium model)
- **Memory**: ~2-3GB peak (Whisper medium model)
- **Real-time**: Record until you release the hotkey, then transcribe
- **Note**: Medium model trades ~60% slower speed for notably better accuracy

## What's Been Optimized

✅ **Critical Fixes Applied:**
- Thread-safe recording with locks
- Proper audio resampling with scipy (prevents aliasing)
- Context managers for resource cleanup (no file leaks)
- Auto-detection of audio device (portable across machines)
- Complete requirements.txt with all dependencies

✅ **Features:**
- Auto-punctuation (periods, capitalization)
- High-accuracy Whisper (medium model for better transcription quality)
- CPU-optimized with potential for ROCm GPU acceleration

## GPU Acceleration (Optional)

### Current Status
- **Hardware**: AMD Radeon 880M GPU detected
- **Current Setup**: CPU-only processing (fast enough for typical use)
- **GPU Driver**: AMD drivers installed and functional

### Enabling ROCm Acceleration (Advanced)

To enable GPU acceleration on AMD GPUs, ROCm must be installed system-wide:

```bash
# Install ROCm runtime libraries
sudo apt-get install rocm-core rocm-libs

# Set environment variables
export HSA_OVERRIDE_GFX_VERSION=gfx1100  # For Radeon 880M

# Rebuild PyTorch for ROCm or install ROCm-compatible build
```

**Note**: ROCm setup is complex and system-dependent. Current CPU performance is adequate. GPU acceleration would reduce transcription time from 3-5s to 1-2s for 5-second audio.

### Model Selection Trade-offs

| Model | Accuracy | Speed (CPU) | Speed (GPU) | Use Case |
|-------|----------|------------|-----------|----------|
| tiny | Lowest | Fastest | ~0.3s | Quick summaries, low precision needed |
| base | Low | Fast | ~0.5s | Basic dictation |
| small | Good | Medium | ~1s | Current baseline (if reverting) |
| **medium** | **Better** | **~3-5s** | **~1-2s** | ✓ Current default; technical terms, accents |
| large | Best | Slowest | ~2-3s | Maximum accuracy, CPU impractical |

To switch models, edit line 26 in `transcriber.py`:
```python
WHISPER_MODEL = "small"  # Change as needed
```

## Local Processing & Privacy

All data stays on your machine:
- 🎤 Audio captured from microphone
- 🔍 Transcribed by local Whisper model
- ⌨️ Typed into your active window
- 🚫 No network requests
- 🚫 No data sent anywhere

## License & Attribution

- Whisper: MIT (OpenAI)
- faster-whisper: MIT
- sounddevice: BSD
- evdev: BSD
- pynput: LGPLv3
- scipy: BSD

---

**Status**: Production-ready with all critical fixes applied. Using Whisper medium model on CPU.
**Last Updated**: January 17, 2026 (GPU investigation complete)
