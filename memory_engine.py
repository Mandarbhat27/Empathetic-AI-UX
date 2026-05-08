"""
Memory Engine — v2
Dual-mode: ChromaDB vector search (primary) + JSON fallback
Runs 100% locally — no cloud, no GPU
Install: pip install chromadb sentence-transformers --break-system-packages
"""

import json
import os
import datetime
from collections import deque

MEMORY_FILE = "memory_store.json"
CHROMA_DIR  = "chroma_memory"

_chroma_client     = None
_chroma_collection = None

def _init_chroma():
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return True
    try:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="empathyos_memory",
            metadata={"hnsw:space": "cosine"}
        )
        return True
    except Exception:
        return False


class MemoryEngine:
    def __init__(self, max_short_term: int = 10, max_long_term: int = 100):
        self.short_term    = deque(maxlen=max_short_term)
        self.long_term     = []
        self.emotion_history = deque(maxlen=30)
        self._entry_count  = 0
        self._use_chroma   = _init_chroma()
        self._load()

    def add(self, text: str, emotion: str):
        self._entry_count += 1
        entry = {
            "id":        str(self._entry_count),
            "text":      text,
            "emotion":   emotion,
            "timestamp": datetime.datetime.now().isoformat(),
            "date":      datetime.datetime.now().strftime("%b %d"),
        }
        self.short_term.append(entry)
        self.long_term.append(entry)
        self.emotion_history.append(emotion)

        if self._use_chroma and _chroma_collection is not None:
            try:
                _chroma_collection.add(
                    documents=[text],
                    metadatas=[{"emotion": emotion, "timestamp": entry["timestamp"]}],
                    ids=[entry["id"]]
                )
            except Exception:
                pass
        self._save_json()

    def get_context(self) -> str:
        if not self.short_term:
            return ""
        return "\n".join(f"[{m['emotion']}] {m['text']}" for m in list(self.short_term)[-5:])

    def search_similar(self, query: str, n_results: int = 3) -> list:
        """Semantic search — finds past conversations similar to current query"""
        if self._use_chroma and _chroma_collection is not None:
            try:
                count = _chroma_collection.count()
                if count == 0:
                    return []
                results = _chroma_collection.query(
                    query_texts=[query],
                    n_results=min(n_results, count)
                )
                return [
                    {
                        "text":      results["documents"][0][i],
                        "emotion":   results["metadatas"][0][i].get("emotion", "neutral"),
                        "timestamp": results["metadatas"][0][i].get("timestamp", ""),
                    }
                    for i in range(len(results["documents"][0]))
                ]
            except Exception:
                pass
        return [
            {"text": e["text"], "emotion": e["emotion"], "timestamp": e["timestamp"]}
            for e in list(self.short_term)[-n_results:]
        ]

    def get_dominant_emotion(self) -> str:
        if not self.emotion_history:
            return "neutral"
        counts = {}
        for e in self.emotion_history:
            counts[e] = counts.get(e, 0) + 1
        return max(counts, key=counts.get)

    def get_emotion_trend(self, n: int = 5) -> list:
        return list(self.emotion_history)[-n:]

    def get_storage_info(self) -> dict:
        return {
            "backend":       "ChromaDB (vector search)" if self._use_chroma else "JSON (keyword)",
            "total_entries": len(self.long_term),
            "short_term":    len(self.short_term),
        }

    def _save_json(self):
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump({
                    "long_term":       self.long_term[-100:],
                    "emotion_history": list(self.emotion_history),
                    "entry_count":     self._entry_count,
                }, f, indent=2)
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, "r") as f:
                    data = json.load(f)
                self.long_term    = data.get("long_term", [])
                self._entry_count = data.get("entry_count", len(self.long_term))
                for e in data.get("emotion_history", []):
                    self.emotion_history.append(e)
                for entry in self.long_term[-10:]:
                    self.short_term.append(entry)
        except Exception:
            pass
