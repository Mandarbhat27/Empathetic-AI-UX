"""
Voice Input Engine — v2
Records mic input and transcribes using speech_recognition (Google free API)
Also analyses basic vocal tone features (speaking rate, pauses)
Install: pip install SpeechRecognition pyaudio --break-system-packages
Windows extra: pip install pipwin && pipwin install pyaudio
"""

import io
import threading
import time

# ─── Check dependencies ─────────────────────────────────────────────────────────
def check_dependencies() -> dict:
    status = {"speech_recognition": False, "pyaudio": False}
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
    return status


def transcribe_audio(audio_file_bytes: bytes, file_format: str = "wav") -> dict:
    """
    Transcribe uploaded audio file bytes to text.
    Returns dict with: text, confidence, error
    """
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        audio_io = io.BytesIO(audio_file_bytes)

        with sr.AudioFile(audio_io) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return {"text": text, "confidence": "high", "error": None}

    except Exception as e:
        error_msg = str(e)
        if "recognition connection failed" in error_msg.lower():
            return {"text": "", "confidence": None, "error": "No internet connection for speech recognition"}
        elif "unintelligible" in error_msg.lower():
            return {"text": "", "confidence": None, "error": "Could not understand audio — please speak clearly"}
        else:
            return {"text": "", "confidence": None, "error": f"Transcription failed: {error_msg}"}


def analyse_vocal_tone(audio_file_bytes: bytes) -> dict:
    """
    Basic vocal tone analysis — speaking rate proxy using audio duration vs word count.
    Returns: speaking_rate (words/min), tone_hint
    """
    try:
        import wave
        import struct
        import math

        audio_io = io.BytesIO(audio_file_bytes)
        with wave.open(audio_io) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration_sec = frames / float(rate)
            raw_data = wf.readframes(frames)
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()

        # Calculate RMS amplitude (loudness proxy)
        if sample_width == 2:
            fmt = f"{len(raw_data) // 2}h"
            samples = struct.unpack(fmt, raw_data)
            if samples:
                rms = math.sqrt(sum(s * s for s in samples) / len(samples))
                normalized_rms = rms / 32768.0
            else:
                normalized_rms = 0.0
        else:
            normalized_rms = 0.5

        return {
            "duration_sec": round(duration_sec, 1),
            "amplitude": round(normalized_rms, 3),
            "tone_hint": _interpret_tone(normalized_rms, duration_sec),
            "error": None
        }
    except Exception as e:
        return {"duration_sec": 0, "amplitude": 0, "tone_hint": "unknown", "error": str(e)}


def _interpret_tone(amplitude: float, duration: float) -> str:
    """Heuristic tone interpretation from audio features"""
    if amplitude > 0.35:
        return "energetic"       # Loud → excited or angry
    elif amplitude > 0.2:
        return "assertive"       # Medium-loud → stressed or confident
    elif amplitude > 0.1:
        return "calm"            # Medium → neutral or happy
    else:
        return "quiet"           # Soft → sad or anxious


def tone_to_emotion_hint(tone: str) -> str:
    """Map tone hint to an emotion modifier"""
    return {
        "energetic": "excited or angry",
        "assertive": "stressed or confident",
        "calm":      "neutral or happy",
        "quiet":     "sad or anxious",
    }.get(tone, "neutral")