#!/usr/bin/env python3
"""
Voice transcriber using OpenAI Whisper and evdev for hotkey activation.
Works with both Wayland and X11. No sudo required with proper udev rules.

Hotkey: Left Shift + Left Ctrl + Space
"""

import os
import sys
import re
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
import atexit
import subprocess

# Save original stderr for restoration
original_stderr = sys.stderr

# Suppress pynput cleanup errors on exit
def _suppress_pynput_exit_errors():
    """Suppress harmless pynput cleanup errors at exit."""
    sys.stderr = open(os.devnull, 'w')

atexit.register(_suppress_pynput_exit_errors)

# Configuration
WHISPER_MODEL = "medium"
SILENCE_TIMEOUT = 1
SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2
MICROPHONE_GAIN = 0.77
AUTO_PUNCTUATION = True
USE_GPU = True
WHISPER_SAMPLE_RATE = 16000

# Recording duration thresholds (seconds)
RECORDING_WARN_THRESHOLD = 30   # Yellow warning
RECORDING_LONG_THRESHOLD = 60   # Red warning

# Audio level targets for normalization reporting
IDEAL_FINAL_PEAK = 0.7  # Target peak after normalization + gain (lower = more amplification)
IDEAL_FINAL_PEAK_MIN = 0.65
IDEAL_FINAL_PEAK_MAX = 0.85
AUDIO_ANALYSIS_VERBOSE = False  # Set to True to see detailed level breakdown

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
        self.device_channels = AUDIO_CHANNELS  # Will be updated by _find_audio_device
        self.current_gain = MICROPHONE_GAIN  # Track current gain for auto-adjustment
        self._setup_whisper()
        self._find_audio_device()  # CRITICAL: Auto-detect
        self._enable_usb_mic_agc()  # Enable AGC on USB mics (fixes silent mic after reconnect)
        self._verify_audio_device()  # Verify device is actually working
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

            # Priority 1: USB devices (usually external mics)
            for idx, device in enumerate(devices):
                if device['max_input_channels'] >= 1 and 'usb' in device['name'].lower():
                    self.audio_device = idx
                    self.device_channels = device['max_input_channels']
                    print(f"✓ Audio device: {idx}: {device['name']} ({device['max_input_channels']}ch)")
                    return

            # Priority 2: Devices with 2+ channels (stereo)
            for idx, device in enumerate(devices):
                if device['max_input_channels'] >= 2:
                    self.audio_device = idx
                    self.device_channels = device['max_input_channels']
                    print(f"✓ Audio device: {idx}: {device['name']} ({device['max_input_channels']}ch)")
                    return

            # Priority 3: Any device with 1+ channel (mono)
            for idx, device in enumerate(devices):
                if device['max_input_channels'] >= 1:
                    self.audio_device = idx
                    self.device_channels = device['max_input_channels']
                    print(f"✓ Audio device: {idx}: {device['name']} ({device['max_input_channels']}ch)")
                    return

            print("❌ No suitable audio device found")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error finding audio device: {e}")
            sys.exit(1)

    def _enable_usb_mic_agc(self):
        """Enable Auto Gain Control on USB microphone if available."""
        if self.audio_device is None:
            return

        try:
            device_info = sd.query_devices(self.audio_device)
            device_name = device_info['name'].lower()

            # Only apply to USB devices
            if 'usb' not in device_name:
                return

            # Extract ALSA card number from device name (e.g., "hw:2,0" -> "2")
            import re
            match = re.search(r'hw:(\d+)', device_info['name'])
            if not match:
                return

            card_num = match.group(1)

            # Enable AGC using amixer
            result = subprocess.run(
                ['amixer', '-c', card_num, 'set', 'Auto Gain Control', 'on'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  AGC enabled on card {card_num}")
        except Exception:
            pass  # AGC not available or already enabled

    def _verify_audio_device(self):
        """Verify the audio device is actually working by doing a test recording."""
        print("Verifying audio device...", end=" ")
        try:
            with suppress_stderr():
                # Record 0.5 seconds and check if we get any audio
                stream = sd.InputStream(
                    device=self.audio_device,
                    samplerate=SAMPLE_RATE,
                    channels=self.device_channels,
                    blocksize=4096,
                    dtype=np.float32
                )
                with stream:
                    data, _ = stream.read(int(SAMPLE_RATE * 0.5))  # 0.5 second

                # Check if device actually captured data (not all zeros)
                if np.abs(data).max() > 0.0001:
                    print("✓")
                    return True
                else:
                    print("✗ (no audio detected)")
                    print("⚠️  Audio device not responding. Try:")
                    print("  1. Unplug microphone, wait 3 seconds, plug back in")
                    print("  2. Run: sudo modprobe -r snd_usb_audio && sudo modprobe snd_usb_audio")
                    print("  3. Restart the transcriber")
                    return False
        except Exception as e:
            print(f"✗ ({e})")
            return False

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

    def _apply_gradual_gain_adjustment(self, recommended_gain):
        """Apply 50% of recommended adjustment gradually, and persist to config."""
        # Only adjust if recommendation differs meaningfully from current (>5%)
        diff_percent = abs(recommended_gain - self.current_gain) / self.current_gain * 100
        if diff_percent < 5:
            return  # Too small to adjust

        # Apply 50% of the difference
        adjustment = (recommended_gain - self.current_gain) * 0.5
        new_gain = self.current_gain + adjustment
        new_gain = max(0.1, min(10.0, new_gain))  # Clamp between 0.1x and 10.0x

        if abs(new_gain - self.current_gain) > 0.01:  # Only report if meaningful change
            print(f"\n🔧 AUTO-ADJUST: Gain {self.current_gain:.2f}x → {new_gain:.2f}x (+{adjustment:+.2f})")
            self.current_gain = new_gain
            self._update_config_gain(new_gain)

    def _update_config_gain(self, new_gain):
        """Update MICROPHONE_GAIN in the config file."""
        try:
            config_path = os.path.abspath(__file__)
            with open(config_path, 'r') as f:
                content = f.read()

            # Replace MICROPHONE_GAIN value
            updated_content = re.sub(
                r'(MICROPHONE_GAIN\s*=\s*)[\d.]+',
                lambda m: f'{m.group(1)}{new_gain:.2f}',
                content
            )

            with open(config_path, 'w') as f:
                f.write(updated_content)
        except Exception as e:
            print(f"⚠️  Could not update config file: {e}")

    def _analyze_audio_levels(self, raw_audio, noise_sample, denoised_audio, final_audio):
        """Analyze and report audio levels with gain recommendations. Returns recommended gain."""
        raw_peak = np.abs(raw_audio).max()
        denoised_peak = np.abs(denoised_audio).max()
        final_peak = np.abs(final_audio).max()

        # Calculate RMS levels (perceived loudness)
        raw_rms = np.sqrt(np.mean(raw_audio ** 2))
        denoised_rms = np.sqrt(np.mean(denoised_audio ** 2))
        final_rms = np.sqrt(np.mean(final_audio ** 2))

        # Calculate SNR (Signal-to-Noise Ratio)
        if noise_sample > 0:
            snr_db = 20 * np.log10(denoised_rms / noise_sample) if denoised_rms > 0 else 0
        else:
            snr_db = float('inf')

        # Determine if current gain is appropriate
        diff_from_ideal = final_peak - IDEAL_FINAL_PEAK
        gain_adjustment_needed = diff_from_ideal / IDEAL_FINAL_PEAK * 100 if IDEAL_FINAL_PEAK > 0 else 0

        # Calculate recommended gain
        # Note: normalization always brings peak to 0.95, so:
        # final_peak = 0.95 * GAIN, therefore GAIN = final_peak / 0.95
        recommended_gain = IDEAL_FINAL_PEAK / 0.95

        # Determine status
        if final_peak < IDEAL_FINAL_PEAK_MIN:
            status = "❌ Too quiet"
        elif final_peak > IDEAL_FINAL_PEAK_MAX:
            status = "⚠️  Too loud"
        else:
            status = "✓ Good"

        # Lean output (default)
        print(f"\n📊 Peak: {final_peak:.3f} | SNR: {snr_db:.1f}dB | {status}")

        # Verbose output (optional)
        if AUDIO_ANALYSIS_VERBOSE:
            print("\n" + "=" * 60)
            print("📊 DETAILED AUDIO LEVEL ANALYSIS")
            print("=" * 60)
            print(f"Raw Audio Peak:           {raw_peak:.4f} (before any processing)")
            print(f"Noise Floor (estimated):  {noise_sample:.4f}")
            print(f"Denoised Peak:            {denoised_peak:.4f}")
            print(f"Final Peak (after gain):  {final_peak:.4f}")
            print(f"\nSignal-to-Noise Ratio:    {snr_db:.1f} dB (denoised signal)")
            print(f"\nIdeal Target Peak:        {IDEAL_FINAL_PEAK:.2f} (range: {IDEAL_FINAL_PEAK_MIN:.2f}-{IDEAL_FINAL_PEAK_MAX:.2f})")
            print(f"Current Peak vs Target:   {diff_from_ideal:+.4f} ({gain_adjustment_needed:+.1f}%)")
            print(f"\nCurrent MICROPHONE_GAIN:  {self.current_gain:.2f}x")
            if final_peak < IDEAL_FINAL_PEAK_MIN or final_peak > IDEAL_FINAL_PEAK_MAX:
                print(f"Recommended MICROPHONE_GAIN: {recommended_gain:.2f}x")
            print("=" * 60)

        return recommended_gain

    def _format_recording_indicator(self, duration):
        """Format recording duration with visual feedback based on length."""
        bars = int(duration / 5)  # One bar per 5 seconds
        bar_str = "█" * min(bars, 12)  # Max 12 bars (60 seconds)

        if duration >= RECORDING_LONG_THRESHOLD:
            return f"  🔴 [{duration:.1f}s] {bar_str} (long recording)"
        elif duration >= RECORDING_WARN_THRESHOLD:
            return f"  🟡 [{duration:.1f}s] {bar_str}"
        else:
            return f"  🟢 [{duration:.1f}s] {bar_str}"

    def record_and_transcribe(self):
        """Record audio and transcribe it using Whisper."""
        # CRITICAL FIX: Thread-safe recording check with lock
        with self.recording_lock:
            if self.is_recording:
                return
            self.is_recording = True

        self.should_stop_recording = False
        print("\n🎤 Recording... (release hotkey to stop)")
        overflow_count = 0  # Track overflows across recording session

        try:
            with suppress_stderr():
                stream = sd.InputStream(
                    device=self.audio_device,
                    samplerate=SAMPLE_RATE,
                    channels=self.device_channels,
                    blocksize=8192,
                    dtype=np.float32,
                    latency="high"  # Allow larger buffers to prevent overflow
                )

                with stream:
                    resampled_audio = []
                    block_count = 0

                    while not self.should_stop_recording:
                        try:
                            data, overflowed = stream.read(8192)
                            if overflowed:
                                overflow_count += 1
                                if overflow_count == 1:
                                    print("⚠️  Audio buffer overflowing - reconnect mic or reduce background noise")
                                # Skip this corrupted block
                                continue

                            # Convert to mono if needed
                            if self.device_channels == 1:
                                mono_data = data.flatten()  # Already mono
                            elif self.device_channels == 2:
                                mono_data = data.mean(axis=1)  # Stereo to mono
                            else:
                                mono_data = data[:, 0]  # Multi-channel, use first channel

                            # CRITICAL FIX: Proper resampling
                            resampled = self._resample_audio(mono_data, SAMPLE_RATE, WHISPER_SAMPLE_RATE)
                            resampled_audio.append(resampled)

                            block_count += 1
                            if block_count % 6 == 0:
                                duration = block_count * 8192 / SAMPLE_RATE
                                indicator = self._format_recording_indicator(duration)
                                print(f"{indicator:<50}", end="\r")

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
                            noise_sample = 0
                            audio_denoised = all_audio

                        # Apply gain
                        amplified = np.clip(normalized * self.current_gain, -0.95, 0.95)
                        audio_int16 = (amplified * 32767).astype(np.int16)

                        # Analyze and report audio levels
                        recommended_gain = self._analyze_audio_levels(all_audio, noise_sample, audio_denoised, amplified)

                        # Auto-adjust gain gradually (50% of difference)
                        self._apply_gradual_gain_adjustment(recommended_gain)

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
        if overflow_count > 0:
            print(f"\n⚠️  Recording had {overflow_count} overflows - audio quality may be degraded")
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
        except OSError as e:
            if e.errno == 19:  # ENODEV - No such device
                print(f"\n✗ Keyboard device disconnected: {e}")
                print("\nYour keyboard was turned off or unplugged.")
                print("Please reconnect it and restart the transcriber.")
            elif e.errno in (13, 1):  # EACCES (13) or EPERM (1) - Permission errors
                print(f"\n✗ Permission error: {e}")
                print("\nYou need to be in the 'input' group to read keyboard events.")
                print("Run these commands:")
                print("  sudo usermod -a -G input $USER")
                print("  # Then log out and log back in")
            else:
                print(f"\n✗ Device error: {e}")
            sys.exit(1)
        except PermissionError as e:
            print(f"\n✗ Permission error: {e}")
            print("\nYou need to be in the 'input' group to read keyboard events.")
            print("Run these commands:")
            print("  sudo usermod -a -G input $USER")
            print("  # Then log out and log back in")
            sys.exit(1)


if __name__ == "__main__":
    transcriber = VoiceTranscriber()
    transcriber.listen_for_hotkey()
