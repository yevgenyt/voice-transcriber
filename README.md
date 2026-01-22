# Voice Transcriber - OpenAI Whisper Edition

Press **Left Shift + Left Ctrl + Space** to speak instructions. Your M6 ZealSound microphone will record, transcribe using faster-Whisper, and type the text directly into your active window.

## Features

- 🎤 **High-accuracy speech recognition** - OpenAI Whisper (medium model) runs offline
- ✍️ **Auto-punctuation** - Automatically adds periods and capitalizes sentences
- ⌨️ **Direct keyboard input** - Types into any active window (uses ydotool on Wayland/GNOME)
- 🚀 **Simple hotkey activation** - Just press a key combination
- 🎯 **Works anywhere** - Compatible with Wayland, X11, any Linux desktop
- ⚡ **CPU optimized** - Fast transcription on CPU; ROCm GPU acceleration available (see notes)
- 🔐 **Privacy-first** - All processing happens locally on your machine
- 📊 **Audio level analysis** - Reports signal quality, noise floor, and SNR after each recording
- 🔧 **Auto-gain adjustment** - Gradually optimizes microphone gain to ideal levels
- 📈 **Visual recording indicator** - Color-coded progress bar shows recording duration
- 🔇 **Noise gate** - Asymmetric attack/release reduces background noise
- 🔌 **Auto-reconnect** - Automatically reconnects if keyboard disconnects
- 🎙️ **AGC auto-enable** - Automatically enables microphone Auto Gain Control

## Quick Start

### 1. Setup (One-Time Only)

Add yourself to the `input` group:
```bash
sudo usermod -a -G input $USER
```

Install ydotool for text input on Wayland/GNOME:
```bash
sudo apt install ydotool
sudo chmod 0666 /dev/uinput  # Required for ydotool
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
                    Noise Gate (5ms attack, 300ms release)
                           ↓
                    scipy.signal (proper resampling to 16kHz)
                           ↓
                    faster-whisper (GPU accelerated)
                           ↓
                    ydotool type (direct text input, no clipboard)
```

## Project Structure

```
~/Applications/voice-transcriber/
├── transcriber.py              ← Main application (entry point)
├── transcriber_config.json     ← User settings (auto-managed)
├── requirements.txt            ← Python dependencies
├── launch.sh                   ← Launcher script
├── README.md                   ← This file
└── venv/                       ← Python virtual environment
```

## Files Overview

| File | Purpose |
|------|---------|
| `transcriber.py` | Main application with all critical fixes |
| `transcriber_config.json` | User settings (microphone_gain) - auto-managed |
| `requirements.txt` | All dependencies with versions |
| `launch.sh` | Simple launcher (aliased as `transcriber`) |
| `venv/` | Virtual environment with installed packages |

## Configuration

### User Settings (transcriber_config.json)

User-adjustable settings are stored in a separate JSON file (auto-created on first run):

```json
{
  "microphone_gain": 0.88
}
```

This file is auto-managed - the gain value is tuned automatically based on audio analysis.

### Application Constants (transcriber.py)

Edit constants in `transcriber.py` to customize behavior:

```python
WHISPER_MODEL = "medium"           # Options: tiny, base, small, medium, large
AUTO_PUNCTUATION = True            # Add periods and capitalization
USE_GPU = True                     # Use GPU if available (ROCm on AMD, CUDA on NVIDIA)
MAX_RECORDING_DURATION = 300       # Safety limit: 5 minutes max recording

# Noise gate settings
NOISE_GATE_VERBOSE = False         # Set True for debug output

# Audio level targets for normalization
IDEAL_FINAL_PEAK = 0.8             # Target peak after processing
IDEAL_FINAL_PEAK_MIN = 0.7         # Too quiet if below this
IDEAL_FINAL_PEAK_MAX = 0.9         # Too loud if above this
```

### Auto-Gain Adjustment

The transcriber automatically tunes `microphone_gain` by analyzing audio levels after each recording:

- **Measures** raw peak, noise floor, denoised signal, and final peak
- **Calculates** SNR (signal-to-noise ratio) to assess audio quality
- **Recommends** ideal gain to reach the target peak (0.8x)
- **Adjusts gradually** by 50% of recommended change per recording
- **Persists** the optimized gain to the config file

**Example output after recording:**
```
============================================================
📊 AUDIO LEVEL ANALYSIS
============================================================
Raw Audio Peak:           0.0675 (before any processing)
Noise Floor (estimated):  0.0003
Denoised Peak:            0.0675
Final Peak (after gain):  0.5685

Signal-to-Noise Ratio:    28.7 dB (denoised signal)

Ideal Target Peak:        0.80 (range: 0.70-0.90)
Current Peak vs Target:   -0.2315 (-28.9%)

Current MICROPHONE_GAIN:  0.84x
✓ Audio level is good!
============================================================
```

**How it works:**
1. After each recording, audio levels are analyzed
2. If levels are off-target, a gain adjustment is recommended
3. Adjustment is applied gradually (50% convergence per step)
4. `MICROPHONE_GAIN` is automatically updated in config
5. Next recording uses the new gain—no restart needed!

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
- Check AGC is enabled: `amixer -c 2 contents | grep -A2 "Auto Gain"`

### Text not typing in applications

ydotool requires uinput access:
```bash
sudo chmod 0666 /dev/uinput
```

Or add yourself to the uinput group:
```bash
sudo groupadd -f uinput
sudo usermod -a -G uinput $USER
echo 'KERNEL=="uinput", GROUP="uinput", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
# Log out and back in
```

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
| Audio Processing | scipy.signal, numpy | Resampling, noise gate |
| Speech Recognition | faster-whisper | Offline transcription |
| Text Input | ydotool | Direct typing on Wayland/GNOME |
| Notifications | notify-send | Desktop notifications |

## Critical Implementation Details

**Audio Pipeline:**
- Records at 48kHz (microphone native rate)
- Applies noise gate with asymmetric timing (5ms attack, 300ms release)
- Resamples to 16kHz using scipy's polyphase filter (anti-aliasing)
- Applies peak normalization
- Feeds to Whisper for transcription

**Thread Safety:**
- Uses `threading.Lock()` for recording state
- Uses `threading.Event()` for stop signaling
- Prevents race conditions on multi-core systems

**Resource Management:**
- Context managers for all file operations
- Proper cleanup of temporary audio files
- Thread cleanup on exit with join timeouts
- No file descriptor leaks

**Keyboard Handling:**
- Auto-reconnect on keyboard disconnect (indefinite wait)
- Desktop notifications via notify-send
- Automatic AGC enabling for USB microphones

**Text Input (Wayland/GNOME):**
- Uses `ydotool type` for direct text input
- No clipboard corruption (doesn't use wl-copy)
- Works in browsers, terminals, and native apps

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
- Thread-safe recording with locks and events
- Proper audio resampling with scipy (prevents aliasing)
- Context managers for resource cleanup (no file leaks)
- Auto-detection of audio device (portable across machines)
- Complete requirements.txt with all dependencies
- Race condition fixes in hotkey handler
- Proper thread cleanup on exit
- Max recording duration (5 min) safety limit

✅ **Audio Quality Features:**
- Auto-punctuation (periods, capitalization)
- High-accuracy Whisper (medium model for better transcription quality)
- Noise gate with asymmetric timing (fast attack, slow release)
- Audio level analysis (reports SNR, noise floor, peak levels)
- Auto-gain adjustment (gradually optimizes microphone gain)
- Auto AGC enabling for USB microphones
- Intelligent thresholds (only adjusts >5% differences, clamps 0.1x-10.0x)
- CPU-optimized with potential for ROCm GPU acceleration

✅ **User Experience:**
- Visual recording indicator with color-coded progress bar
- Keyboard auto-reconnect with desktop notifications
- Direct text input via ydotool (no clipboard corruption)
- JSON config file for user settings (no source code modification)

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

**Status**: Production-ready with noise gate, visual indicator, auto-reconnect, and ydotool text input.
**Last Updated**: January 21, 2026 (Major refactor: noise gate, visual indicator, code quality fixes, ydotool support)
