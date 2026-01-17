#!/usr/bin/env python3
"""
Voice transcriber using OpenAI Whisper and evdev for hotkey activation.
Works with both Wayland and X11. No sudo required with proper udev rules.

Hotkey: Left Shift + Left Ctrl + Space
"""

import os
import sys
import sounddevice as sd
import numpy as np
import soundfile as sf
import scipy.signal
import threading
from contextlib import contextmanager
from faster_whisper import WhisperModel
from pynput.keyboard import Controller
from evdev import InputDevice, list_devices, ecodes
import tempfile

# Save original stderr for restoration
original_stderr = sys.stderr

# Configuration
WHISPER_MODEL = "medium"
SILENCE_TIMEOUT = 1
SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2
MICROPHONE_GAIN = 2.0
AUTO_PUNCTUATION = True
USE_GPU = True
WHISPER_SAMPLE_RATE = 16000

# Hotkey codes
KEY_LEFTSHIFT = ecodes.KEY_LEFTSHIFT
KEY_LEFTCTRL = ecodes.KEY_LEFTCTRL
KEY_SPACE = ecodes.KEY_SPACE


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr without resource leaks."""
    old_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


class VoiceTranscriber:
    def __init__(self):
        self.keyboard_controller = Controller()
        self.is_recording = False
        self.should_stop_recording = False
        self.recording_lock = threading.Lock()  # CRITICAL: Thread safety
        self.pressed_keys = set()
        self.model = None
        self.keyboard_device = None
        self.audio_device = None
        self._setup_whisper()
        self._find_audio_device()  # CRITICAL: Auto-detect
        self._find_keyboard_device()

    def _setup_whisper(self):
        """Initialize Whisper model using faster-whisper."""
        print("Initializing Whisper (faster-whisper)...")

        # Detect GPU
        import torch
        gpu_available = torch.cuda.is_available()
        device = "cuda" if (gpu_available and USE_GPU) else "cpu"
        compute_type = "float16" if (gpu_available and USE_GPU) else "int8"

        if gpu_available and USE_GPU:
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  Using CPU (GPU not available or disabled)")

        try:
            with suppress_stderr():
                self.model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
            print(f"✓ Whisper {WHISPER_MODEL} initialized on {device.upper()}")
        except Exception as e:
            print(f"❌ Error loading Whisper: {e}")
            sys.exit(1)

    def _find_audio_device(self):
        """Auto-detect audio input device."""
        print("Finding audio device...")

        try:
            devices = sd.query_devices()
            for idx, device in enumerate(devices):
                if device['max_input_channels'] >= 2 and 'mini' in device['name'].lower():
                    self.audio_device = idx
                    print(f"✓ Audio device: {idx}: {device['name']}")
                    return

            # Fallback: find any device with 2+ input channels
            for idx, device in enumerate(devices):
                if device['max_input_channels'] >= 2:
                    self.audio_device = idx
                    print(f"✓ Audio device: {idx}: {device['name']}")
                    return

            print("❌ No suitable audio device found")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error finding audio device: {e}")
            sys.exit(1)

    def _find_keyboard_device(self):
        """Find and open the keyboard device using evdev."""
        print("Finding keyboard device...")

        try:
            devices = list_devices()
            if not devices:
                print("✗ No input devices found. You may need to add your user to the 'input' group.")
                print("\nTo fix this, run:")
                print("  sudo usermod -a -G input $USER")
                print("  # Then log out and log back in")
                sys.exit(1)

            keyboard_found = False
            for device_path in devices:
                try:
                    device = InputDevice(device_path)
                    if ecodes.EV_KEY in device.capabilities():
                        keys = device.capabilities()[ecodes.EV_KEY]
                        if ecodes.KEY_A in keys and ecodes.KEY_SPACE in keys:
                            print(f"✓ Keyboard found: {device.name}")
                            print(f"  Device path: {device_path}")
                            self.keyboard_device = device
                            keyboard_found = True
                            break
                except (OSError, PermissionError):
                    pass

            if not keyboard_found:
                print("✗ Could not find a usable keyboard device.")
                print("\nTroubleshooting:")
                print("1. Make sure you're in the 'input' group:")
                print("   groups | grep input")
                sys.exit(1)

        except Exception as e:
            print(f"✗ Error finding keyboard device: {e}")
            sys.exit(1)

    def _resample_audio(self, audio, orig_sr, target_sr):
        """CRITICAL FIX: Proper audio resampling with anti-aliasing."""
        if orig_sr == target_sr:
            return audio
        # Use scipy's polyphase resampler (prevents aliasing)
        return scipy.signal.resample_poly(audio, target_sr, orig_sr)

    def record_and_transcribe(self):
        """Record audio and transcribe it using Whisper."""
        # CRITICAL FIX: Thread-safe recording check with lock
        with self.recording_lock:
            if self.is_recording:
                return
            self.is_recording = True

        self.should_stop_recording = False
        print("\n🎤 Recording... (release hotkey to stop)")

        try:
            with suppress_stderr():
                stream = sd.InputStream(
                    device=self.audio_device,
                    samplerate=SAMPLE_RATE,
                    channels=AUDIO_CHANNELS,
                    blocksize=4096,
                    dtype=np.float32
                )

                with stream:
                    resampled_audio = []
                    block_count = 0

                    while not self.should_stop_recording:
                        try:
                            data, overflowed = stream.read(4096)
                            if overflowed:
                                print("⚠️  Audio overflow detected")

                            # Convert stereo to mono
                            if AUDIO_CHANNELS == 2:
                                mono_data = data.mean(axis=1)
                            else:
                                mono_data = data.flatten()

                            # CRITICAL FIX: Proper resampling
                            resampled = self._resample_audio(mono_data, SAMPLE_RATE, WHISPER_SAMPLE_RATE)
                            resampled_audio.append(resampled)

                            block_count += 1
                            if block_count % 12 == 0:
                                duration = block_count * 4096 / SAMPLE_RATE
                                print(f"  [{duration:.1f}s recorded]", end="\r")

                        except Exception as e:
                            print(f"⚠️  Error recording: {e}")
                            continue

                    # Process and normalize audio
                    if resampled_audio:
                        all_audio = np.concatenate(resampled_audio)
                        peak = np.abs(all_audio).max()

                        if peak > 0:
                            # Boundary check: CRITICAL FIX
                            if len(resampled_audio) >= 10:
                                noise_sample = np.abs(np.concatenate(resampled_audio[:len(resampled_audio)//10])).mean()
                            else:
                                noise_sample = np.abs(all_audio[:len(all_audio)//10]).mean()

                            audio_denoised = all_audio.copy()
                            audio_denoised[np.abs(audio_denoised) < noise_sample * 1.5] = 0

                            peak_denoised = np.abs(audio_denoised).max()
                            if peak_denoised > 0:
                                normalized = (audio_denoised / peak_denoised) * 0.95
                            else:
                                normalized = all_audio / peak * 0.9
                        else:
                            normalized = all_audio

                        # Apply gain
                        amplified = np.clip(normalized * MICROPHONE_GAIN, -0.95, 0.95)
                        audio_int16 = (amplified * 32767).astype(np.int16)

                        # Save to temp file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            with sf.SoundFile(tmp.name, 'w', WHISPER_SAMPLE_RATE, 1, 'PCM_16') as f:
                                f.write(audio_int16)
                            temp_path = tmp.name
                    else:
                        temp_path = None

        except Exception as e:
            print(f"⚠️  Recording error: {e}")
            with self.recording_lock:
                self.is_recording = False
            return

        # Transcribe
        print("\n⏳ Transcribing...")
        transcribed_text = self._transcribe_audio(temp_path)

        print(" " * 100, end="\r")

        if transcribed_text:
            print(f"📝 Transcribed: \"{transcribed_text}\"")
            self.keyboard_controller.type(transcribed_text + " ")
            print("✓ Text typed into active window\n")
        else:
            print("❌ No speech detected\n")

        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

        with self.recording_lock:
            self.is_recording = False

    def _apply_punctuation(self, text):
        """Apply auto-punctuation: capitalize and add period."""
        if not text:
            return text

        text = text[0].upper() + text[1:] if len(text) > 0 else text

        if text and text[-1] not in ".!?":
            text += "."

        return text

    def _transcribe_audio(self, audio_path):
        """Transcribe audio file using faster-whisper."""
        if not audio_path:
            return ""

        try:
            with suppress_stderr():
                print("  Processing with Whisper...", end="\r")
                segments, info = self.model.transcribe(audio_path, language="en", beam_size=5)
                text = " ".join([segment.text for segment in segments]).strip()

            if AUTO_PUNCTUATION and text:
                text = self._apply_punctuation(text)

            return text
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""

    def listen_for_hotkey(self):
        """Listen for hotkey using evdev."""
        print("=" * 50)
        print("🎙️  Voice Transcriber Started")
        print("=" * 50)
        print(f"Hotkey: Left Shift + Left Ctrl + Space")
        print(f"Model: Whisper {WHISPER_MODEL}")
        print("Features: Auto-punctuation\n")
        print("Press Ctrl+C to exit\n")

        try:
            while True:
                for event in self.keyboard_device.read_loop():
                    if event.type != ecodes.EV_KEY:
                        continue

                    key_event = event.value

                    if key_event == 1:  # Key press
                        if event.code == KEY_LEFTSHIFT:
                            self.pressed_keys.add(KEY_LEFTSHIFT)
                        elif event.code == KEY_LEFTCTRL:
                            self.pressed_keys.add(KEY_LEFTCTRL)
                        elif event.code == KEY_SPACE:
                            if KEY_LEFTSHIFT in self.pressed_keys and KEY_LEFTCTRL in self.pressed_keys:
                                if not self.is_recording:
                                    threading.Thread(target=self.record_and_transcribe, daemon=False).start()

                    elif key_event == 0:  # Key release
                        if event.code == KEY_LEFTSHIFT:
                            self.pressed_keys.discard(KEY_LEFTSHIFT)
                        elif event.code == KEY_LEFTCTRL:
                            self.pressed_keys.discard(KEY_LEFTCTRL)
                        elif event.code == KEY_SPACE:
                            if self.is_recording:
                                self.should_stop_recording = True

        except KeyboardInterrupt:
            print("\n\n👋 Transcriber stopped")
            sys.exit(0)
        except (OSError, PermissionError) as e:
            print(f"\n✗ Permission error: {e}")
            print("\nYou need to be in the 'input' group to read keyboard events.")
            print("Run these commands:")
            print("  sudo usermod -a -G input $USER")
            sys.exit(1)


if __name__ == "__main__":
    transcriber = VoiceTranscriber()
    transcriber.listen_for_hotkey()
