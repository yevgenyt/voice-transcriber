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
import json
import time

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

# Noise gate settings (asymmetric attack/release)
NOISE_GATE_ATTACK_MS = 5        # Fast attack (ms) - quickly respond to speech
NOISE_GATE_RELEASE_MS = 300     # Slow release (ms) - gradually fade during pauses
NOISE_GATE_THRESHOLD = 2.0      # Multiplier of noise floor to trigger gate open
NOISE_GATE_REDUCTION = 0.1      # Gain reduction when gate is closed (0.1 = -20dB)
NOISE_GATE_VERBOSE = False      # Set to True to see noise gate metrics after each recording

# Keyboard reconnection settings
KEYBOARD_RECONNECT_DELAY = 3        # Seconds between reconnection attempts

# Recording limits
MAX_RECORDING_DURATION = 300        # Maximum recording duration in seconds (5 minutes)

# Config file path
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcriber_config.json")

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
    devnull = None
    try:
        devnull = open(os.devnull, 'w')
        sys.stderr = devnull
        yield
    finally:
        sys.stderr = old_stderr
        if devnull is not None:
            devnull.close()


class VoiceTranscriber:
    def __init__(self):
        self.keyboard_controller = Controller()
        self.is_recording = False
        self.stop_recording_event = threading.Event()  # Thread-safe stop signal
        self.recording_lock = threading.Lock()  # CRITICAL: Thread safety
        self.recording_thread = None  # Track current recording thread
        self.pressed_keys = set()
        self.model = None
        self.keyboard_device = None
        self.audio_device = None
        self.device_channels = AUDIO_CHANNELS  # Will be updated by _find_audio_device
        self.current_gain = MICROPHONE_GAIN  # Track current gain for auto-adjustment
        self._load_config()  # Load user config from JSON file
        self._setup_whisper()
        self._find_audio_device()  # CRITICAL: Auto-detect
        self._enable_usb_mic_agc()  # Enable AGC on USB mics
        self._verify_audio_device()  # Verify device is working
        self._find_keyboard_device()

    def _load_config(self):
        """Load user configuration from JSON file."""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.current_gain = config.get('microphone_gain', MICROPHONE_GAIN)
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")

    def _save_config(self):
        """Save user configuration to JSON file."""
        try:
            config = {
                'microphone_gain': self.current_gain
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save config: {e}")
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
        """Update microphone gain in the config file."""
        self._save_config()

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

    def _apply_noise_gate(self, audio, sample_rate, noise_floor):
        """
        Apply noise gate with fast attack / slow release to reduce AGC pumping artifacts.
        Returns (processed_audio, metrics_dict).
        """
        # Convert time constants to samples
        attack_samples = int(NOISE_GATE_ATTACK_MS * sample_rate / 1000)
        release_samples = int(NOISE_GATE_RELEASE_MS * sample_rate / 1000)

        # Compute envelope using RMS in small windows
        window_size = max(int(sample_rate * 0.01), 1)  # 10ms windows
        envelope = np.zeros(len(audio))

        for i in range(0, len(audio), window_size):
            end = min(i + window_size, len(audio))
            rms = np.sqrt(np.mean(audio[i:end] ** 2))
            envelope[i:end] = rms

        # Calculate metrics BEFORE processing
        threshold = noise_floor * NOISE_GATE_THRESHOLD
        before_silence_energy = np.mean(envelope[envelope < threshold] ** 2) if np.any(envelope < threshold) else 0
        before_speech_energy = np.mean(envelope[envelope >= threshold] ** 2) if np.any(envelope >= threshold) else 0

        # Apply asymmetric smoothing to envelope
        smoothed = np.zeros(len(envelope))
        smoothed[0] = envelope[0]

        for i in range(1, len(envelope)):
            if envelope[i] > smoothed[i-1]:
                # Attack: signal increasing - respond quickly
                alpha = 1.0 - np.exp(-1.0 / max(attack_samples, 1))
            else:
                # Release: signal decreasing - respond slowly
                alpha = 1.0 - np.exp(-1.0 / max(release_samples, 1))
            smoothed[i] = alpha * envelope[i] + (1 - alpha) * smoothed[i-1]

        # Create gain curve: full gain above threshold, reduced below
        gain_curve = np.ones(len(audio))
        below_threshold = smoothed < threshold
        # Soft knee: gradual transition
        transition_zone = (smoothed >= threshold * 0.5) & (smoothed < threshold)
        gain_curve[below_threshold & ~transition_zone] = NOISE_GATE_REDUCTION
        # Smooth transition in the knee
        if np.any(transition_zone):
            t = (smoothed[transition_zone] - threshold * 0.5) / (threshold * 0.5)
            gain_curve[transition_zone] = NOISE_GATE_REDUCTION + t * (1.0 - NOISE_GATE_REDUCTION)

        # Apply gain curve
        processed = audio * gain_curve

        # Calculate metrics AFTER processing
        proc_envelope = np.abs(processed)
        after_silence_energy = np.mean(proc_envelope[below_threshold] ** 2) if np.any(below_threshold) else 0
        after_speech_energy = np.mean(proc_envelope[~below_threshold] ** 2) if np.any(~below_threshold) else 0

        metrics = {
            'silence_reduction_db': 10 * np.log10(after_silence_energy / before_silence_energy) if before_silence_energy > 0 and after_silence_energy > 0 else 0,
            'speech_preserved_pct': 100 * after_speech_energy / before_speech_energy if before_speech_energy > 0 else 100,
            'gate_active_pct': 100 * np.sum(below_threshold) / len(below_threshold),
        }

        return processed, metrics

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

    def _record_audio(self):
        """Record audio from microphone. Returns (audio_data, overflow_count) or (None, 0) on error."""
        overflow_count = 0
        resampled_audio = []

        try:
            with suppress_stderr():
                stream = sd.InputStream(
                    device=self.audio_device,
                    samplerate=SAMPLE_RATE,
                    channels=self.device_channels,
                    blocksize=8192,
                    dtype=np.float32,
                    latency="high"
                )

                with stream:
                    block_count = 0
                    max_blocks = int(MAX_RECORDING_DURATION * SAMPLE_RATE / 8192)

                    while not self.stop_recording_event.is_set() and block_count < max_blocks:
                        try:
                            data, overflowed = stream.read(8192)
                            if overflowed:
                                overflow_count += 1
                                if overflow_count == 1:
                                    print("⚠️  Audio buffer overflowing")
                                continue

                            # Convert to mono
                            if self.device_channels == 1:
                                mono_data = data.flatten()
                            elif self.device_channels == 2:
                                mono_data = data.mean(axis=1)
                            else:
                                mono_data = data[:, 0]

                            resampled = self._resample_audio(mono_data, SAMPLE_RATE, WHISPER_SAMPLE_RATE)
                            resampled_audio.append(resampled)

                            block_count += 1
                            if block_count % 6 == 0:
                                duration = block_count * 8192 / SAMPLE_RATE
                                indicator = self._format_recording_indicator(duration)
                                print(f"{indicator:<50}", end="\r")

                            # Check for max duration
                            if block_count >= max_blocks:
                                print(f"\n⚠️  Max recording duration ({MAX_RECORDING_DURATION}s) reached")

                        except Exception as e:
                            print(f"⚠️  Error recording: {e}")
                            continue

            if resampled_audio:
                return np.concatenate(resampled_audio), overflow_count
            return None, overflow_count

        except Exception as e:
            print(f"⚠️  Recording error: {e}")
            return None, overflow_count

    def _process_audio(self, all_audio):
        """Process recorded audio: noise gate, normalization, gain. Returns processed audio or None."""
        if all_audio is None or len(all_audio) == 0:
            return None, 0

        peak = np.abs(all_audio).max()
        if peak == 0:
            return all_audio, 0

        # Calculate noise floor with safety check for short recordings
        min_samples = max(160, len(all_audio) // 10)  # At least 10ms or 10% of audio
        if len(all_audio) < min_samples:
            noise_sample = np.abs(all_audio).mean()
        else:
            noise_sample = np.abs(all_audio[:min_samples]).mean()

        # Ensure noise_sample is valid
        if noise_sample == 0 or np.isnan(noise_sample):
            noise_sample = 0.001  # Fallback to small value

        # Apply noise gate
        gated_audio, gate_metrics = self._apply_noise_gate(all_audio, WHISPER_SAMPLE_RATE, noise_sample)

        if NOISE_GATE_VERBOSE:
            print(f"\n🔇 Noise gate: silence {gate_metrics['silence_reduction_db']:.1f}dB | "
                  f"speech {gate_metrics['speech_preserved_pct']:.0f}% preserved | "
                  f"gate active {gate_metrics['gate_active_pct']:.0f}%")

        # Normalize (noise gate already handled noise reduction, no need for hard threshold)
        peak_gated = np.abs(gated_audio).max()
        if peak_gated > 0:
            normalized = (gated_audio / peak_gated) * 0.95
        else:
            normalized = gated_audio

        # Apply gain
        amplified = np.clip(normalized * self.current_gain, -0.95, 0.95)

        # Analyze and auto-adjust
        recommended_gain = self._analyze_audio_levels(all_audio, noise_sample, gated_audio, amplified)
        self._apply_gradual_gain_adjustment(recommended_gain)

        return amplified, noise_sample

    def record_and_transcribe(self):
        """Record audio and transcribe it using Whisper."""
        self.stop_recording_event.clear()
        print("\n🎤 Recording... (release hotkey to stop)")

        # Record audio
        all_audio, overflow_count = self._record_audio()

        if all_audio is None or len(all_audio) == 0:
            print("❌ No audio recorded\n")
            with self.recording_lock:
                self.is_recording = False
            return

        # Process audio
        amplified, noise_sample = self._process_audio(all_audio)

        if amplified is None:
            print("❌ Audio processing failed\n")
            with self.recording_lock:
                self.is_recording = False
            return

        # Save to temp file
        audio_int16 = (amplified * 32767).astype(np.int16)
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                with sf.SoundFile(tmp.name, 'w', WHISPER_SAMPLE_RATE, 1, 'PCM_16') as f:
                    f.write(audio_int16)
                temp_path = tmp.name
        except Exception as e:
            print(f"⚠️  Error saving audio: {e}")
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
            if self._type_text(transcribed_text + " "):
                print("✓ Text typed into active window\n")
            else:
                print("❌ Failed to type text\n")
        else:
            print("❌ No speech detected\n")

        # Cleanup temp file
        if os.path.exists(temp_path):
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

    def _type_text(self, text):
        """Type text directly using ydotool (doesn't use clipboard)."""
        try:
            # Type text directly - doesn't corrupt clipboard
            subprocess.run(["ydotool", "type", "--", text], capture_output=True, timeout=10)
            return True
        except FileNotFoundError:
            # Fallback to pynput if ydotool not available
            self.keyboard_controller.type(text)
            return True
        except Exception as e:
            print(f"⚠️  Typing failed: {e}")
            try:
                self.keyboard_controller.type(text)
                return True
            except Exception:
                return False

    def _cleanup(self):
        """Clean up resources before exit."""
        self.stop_recording_event.set()
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=3)
        if self.keyboard_device:
            try:
                self.keyboard_device.close()
            except Exception:
                pass

    def _reset_recording_state(self):
        """Reset all recording state (used after keyboard reconnection)."""
        self.pressed_keys.clear()
        self.stop_recording_event.set()  # Stop any ongoing recording
        with self.recording_lock:
            self.is_recording = False
        # Wait for recording thread to finish if running
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2)
        self.recording_thread = None

    def _send_notification(self, title, message, urgency="critical"):
        """Send desktop notification."""
        try:
            subprocess.run(
                ["notify-send", "-u", urgency, "-a", "Voice Transcriber", title, message],
                capture_output=True, timeout=2
            )
        except Exception:
            pass  # Notification not available

    def _reconnect_keyboard(self):
        """Attempt to reconnect to keyboard after disconnection. Waits indefinitely."""
        print("\n⏳ Waiting for keyboard to reconnect (Ctrl+C to exit)...")
        self._send_notification("⌨️ Keyboard Disconnected", "Waiting for keyboard to reconnect...")

        # Flashing indicator characters
        flash_chars = ["⚠️ ", "   "]
        attempt = 0

        while True:
            attempt += 1
            elapsed_min = (attempt * KEYBOARD_RECONNECT_DELAY) // 60
            elapsed_sec = (attempt * KEYBOARD_RECONNECT_DELAY) % 60

            # Flash the warning
            flash = flash_chars[attempt % 2]
            print(f"\r{flash}Keyboard disconnected - waiting {elapsed_min}m {elapsed_sec:02d}s  ", end="", flush=True)

            time.sleep(KEYBOARD_RECONNECT_DELAY)

            try:
                devices = list_devices()
                for device_path in devices:
                    try:
                        device = InputDevice(device_path)
                        if ecodes.EV_KEY in device.capabilities():
                            keys = device.capabilities()[ecodes.EV_KEY]
                            if ecodes.KEY_A in keys and ecodes.KEY_SPACE in keys:
                                self.keyboard_device = device
                                self._reset_recording_state()  # Reset all recording state
                                print(f"\r✓ Keyboard reconnected: {device.name}                              ")
                                self._send_notification("✓ Keyboard Reconnected", device.name, urgency="normal")
                                return True
                    except (OSError, PermissionError):
                        pass
            except Exception:
                pass

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
                                # Fix race condition: check and set inside lock
                                with self.recording_lock:
                                    if not self.is_recording:
                                        self.is_recording = True
                                        self.recording_thread = threading.Thread(
                                            target=self.record_and_transcribe, daemon=True
                                        )
                                        self.recording_thread.start()

                    elif key_event == 0:  # Key release
                        if event.code == KEY_LEFTSHIFT:
                            self.pressed_keys.discard(KEY_LEFTSHIFT)
                        elif event.code == KEY_LEFTCTRL:
                            self.pressed_keys.discard(KEY_LEFTCTRL)
                        elif event.code == KEY_SPACE:
                            if self.is_recording:
                                self.stop_recording_event.set()  # Thread-safe stop signal

        except KeyboardInterrupt:
            print("\n\n👋 Transcriber stopped")
            self._cleanup()
            sys.exit(0)
        except OSError as e:
            if e.errno == 19:  # ENODEV - No such device
                print(f"\n⚠️  Keyboard disconnected")
                self._reconnect_keyboard()  # Waits indefinitely until reconnected
                print("Resuming hotkey listener...\n")
                self.listen_for_hotkey()  # Restart the listener
            elif e.errno in (13, 1):  # EACCES (13) or EPERM (1) - Permission errors
                print(f"\n✗ Permission error: {e}")
                print("\nYou need to be in the 'input' group to read keyboard events.")
                print("Run these commands:")
                print("  sudo usermod -a -G input $USER")
                print("  # Then log out and log back in")
                sys.exit(1)
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
