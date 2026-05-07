"""
Memory Engine
Stores conversation history and emotional context locally
No cloud, no GPU — pure Python + JSON
"""

import json
import os
import datetime
from collections import deque

MEMORY_FILE = "memory_store.json"

class MemoryEngine:
    def __init__(self, max_short_term=10, max_long_term=50):
        self.short_term = deque(maxlen=max_short_term)  # last 10 messages
        self.long_term = []                              # persistent storage
        self.emotion_history = deque(maxlen=20)
        self._load()

    def add(self, text: str, emotion: str):
        entry = {
            "text": text,
            "emotion": emotion,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.short_term.append(entry)
        self.long_term.append(entry)
        self.emotion_history.append(emotion)
        self._save()

    def get_context(self) -> str:
        """Build a context string for the LLM from recent memory"""
        if not self.short_term:
            return ""
        lines = []
        for m in list(self.short_term)[-5:]:
            lines.append(f"[{m['emotion']}] {m['text']}")
        return "\n".join(lines)

    def get_dominant_emotion(self) -> str:
        if not self.emotion_history:
            return "neutral"
        counts = {}
        for e in self.emotion_history:
            counts[e] = counts.get(e, 0) + 1
        return max(counts, key=counts.get)

    def _save(self):
        try:
            data = {
                "long_term": self.long_term[-50:],
                "emotion_history": list(self.emotion_history)
            }
            with open(MEMORY_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, "r") as f:
                    data = json.load(f)
                self.long_term = data.get("long_term", [])
                for e in data.get("emotion_history", []):
                    self.emotion_history.append(e)
                # Restore last 10 into short term
                for entry in self.long_term[-10:]:
                    self.short_term.append(entry)
        except Exception:
            pass
