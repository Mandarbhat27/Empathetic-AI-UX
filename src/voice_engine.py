"""
Voice Input Engine — v2 (updated)
Handles WAV, MP3, OGG, M4A by auto-converting to WAV first
Install: pip install SpeechRecognition pydub
Windows extra for MP3/OGG: pip install pydub and install ffmpeg
"""

import io
import wave
import struct
import math


def check_dependencies() -> dict:
    status = {"speech_recognition": False, "pyaudio": False, "pydub": False}
    try:
        import speech_recognition
        status["speech_recognition"] = True
    except ImportError:
        pass
    try:
        import pyaudio
        status["pyaudio"] = True
    except ImportError:
        pass
    try:
        import pydub
        status["pydub"] = True
    except ImportError:
        pass
    return status


def _convert_to_wav(audio_bytes: bytes, file_ext: str = "ogg") -> bytes:
    """
    Convert any audio format to WAV using pydub.
    Falls back to raw bytes if pydub not available.
    """
    try:
        from pydub import AudioSegment
        audio_io = io.BytesIO(audio_bytes)

        fmt = file_ext.lower().strip(".")
        if fmt == "m4a":
            fmt = "mp4"  # pydub reads m4a as mp4

        audio = AudioSegment.from_file(audio_io, format=fmt)

        # Convert to mono 16kHz WAV — optimal for speech recognition
        audio = audio.set_channels(1).set_frame_rate(16000)

        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        return wav_io.getvalue()

    except ImportError:
        # pydub not installed — return as-is and let SpeechRecognition try
        return audio_bytes
    except Exception as e:
        return audio_bytes


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.ogg") -> dict:
    """
    Transcribe audio file bytes to text.
    Auto-converts OGG/MP3/M4A to WAV before transcription.
    Returns dict with: text, confidence, error
    """
    try:
        import speech_recognition as sr

        # Detect format from filename
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav"

        # Convert to WAV if not already WAV
        if ext != "wav":
            wav_bytes = _convert_to_wav(audio_bytes, ext)
        else:
            wav_bytes = audio_bytes

        recognizer = sr.Recognizer()
        audio_io   = io.BytesIO(wav_bytes)

        with sr.AudioFile(audio_io) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return {"text": text, "confidence": "high", "error": None}

    except ImportError:
        return {"text": "", "confidence": None, "error": "SpeechRecognition not installed — run: pip install SpeechRecognition"}
    except Exception as e:
        error_msg = str(e)
        if "unintelligible" in error_msg.lower():
            return {"text": "", "confidence": None, "error": "Could not understand audio — please speak clearly and reduce background noise"}
        elif "connection" in error_msg.lower():
            return {"text": "", "confidence": None, "error": "No internet connection — speech recognition needs internet"}
        elif "PCM WAV" in error_msg or "AIFF" in error_msg:
            return {"text": "", "confidence": None, "error": "Format conversion failed — install pydub: pip install pydub  (also needs ffmpeg for OGG/MP3)"}
        else:
            return {"text": "", "confidence": None, "error": f"Transcription failed: {error_msg}"}


def analyse_vocal_tone(audio_bytes: bytes) -> dict:
    """
    Basic vocal tone analysis from WAV amplitude.
    Returns: duration_sec, amplitude, tone_hint
    """
    try:
        audio_io = io.BytesIO(audio_bytes)
        with wave.open(audio_io) as wf:
            frames       = wf.getnframes()
            rate         = wf.getframerate()
            sample_width = wf.getsampwidth()
            duration_sec = frames / float(rate)
            raw_data     = wf.readframes(frames)

        if sample_width == 2:
            fmt     = f"{len(raw_data) // 2}h"
            samples = struct.unpack(fmt, raw_data)
            if samples:
                rms            = math.sqrt(sum(s * s for s in samples) / len(samples))
                normalized_rms = rms / 32768.0
            else:
                normalized_rms = 0.0
        else:
            normalized_rms = 0.5

        return {
            "duration_sec": round(duration_sec, 1),
            "amplitude":    round(normalized_rms, 3),
            "tone_hint":    _interpret_tone(normalized_rms),
            "error":        None
        }
    except Exception as e:
        return {"duration_sec": 0, "amplitude": 0, "tone_hint": "unknown", "error": str(e)}


def _interpret_tone(amplitude: float) -> str:
    if   amplitude > 0.35: return "energetic"
    elif amplitude > 0.2:  return "assertive"
    elif amplitude > 0.1:  return "calm"
    else:                  return "quiet"


def tone_to_emotion_hint(tone: str) -> str:
    return {
        "energetic": "excited or angry",
        "assertive": "stressed or confident",
        "calm":      "neutral or happy",
        "quiet":     "sad or anxious",
    }.get(tone, "neutral")