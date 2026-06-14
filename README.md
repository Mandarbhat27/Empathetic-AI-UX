# 🧠 EmpathyOS — Empathetic AI UX

> An on-device AI companion that detects your emotions, responds empathetically, remembers past conversations, and helps you track your emotional wellbeing — all from a beautiful Streamlit interface.

---

## ✨ What It Does

EmpathyOS is a **mental wellness AI assistant** powered by DistilBERT emotion detection and an LLM-backed empathetic chat engine. You type (or speak) how you feel, and the app:

- Detects your emotion in real time (happy, sad, angry, anxious, stressed, excited, neutral)
- Dynamically changes the UI theme/color to match your mood
- Responds with empathetic, context-aware messages using an LLM
- Remembers your past emotions using **ChromaDB** semantic memory
- Tracks your mood history with streaks, analytics, and timelines
- Suggests personalized wellness techniques based on your current emotion
- Exports a downloadable **PDF mood report**
- Supports both **text** and **voice input** (audio file upload + transcription)

---

## 🏗️ Project Structure

```
Empathetic-AI-UX/
│
├── app.py               # Main Streamlit app — UI, routing, session state
├── emotion_engine.py    # DistilBERT-based emotion detection + keyword fallback
├── llm_engine.py        # LLM integration for empathetic response generation
├── memory_engine.py     # ChromaDB semantic memory + JSON fallback
├── voice_engine.py      # Audio transcription + vocal tone analysis
├── pdf_engine.py        # PDF mood report generation (ReportLab)
├── memory_store.json    # JSON fallback memory store
├── chroma_memory/       # ChromaDB vector database files
├── src/                 # Additional source modules
├── requirements.txt     # All Python dependencies
└── .gitignore
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Emotion Detection | DistilBERT (`transformers`) + keyword fallback |
| LLM / Chat | LLM engine (configurable via `.env`) |
| Semantic Memory | ChromaDB + sentence-transformers |
| Voice Input | SpeechRecognition + optional PyAudio |
| PDF Export | ReportLab |
| Config | python-dotenv |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Mandarbhat27/Empathetic-AI-UX.git
cd Empathetic-AI-UX
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ If you're on a system-managed Python (e.g. Ubuntu 22+), add `--break-system-packages`:
> ```bash
> pip install -r requirements.txt --break-system-packages
> ```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# Add your LLM API key here (e.g. OpenAI, Groq, etc.)
OPENAI_API_KEY=your_api_key_here
```

Check `llm_engine.py` to confirm which LLM provider is configured.

### 5. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🎤 Voice Input Setup

Voice input works via **audio file upload** (WAV, MP3, OGG, M4A) out of the box.

For **live microphone recording**, install PyAudio separately:

```bash
# Standard:
pip install pyaudio

# Windows (if the above fails):
pip install pipwin
pipwin install pyaudio
```

**How to record audio for upload:**
- 📱 iPhone/Android: Use the Voice Memos app → export as M4A or WAV
- 🪟 Windows: Open "Voice Recorder" → save as MP3
- 🍎 Mac: Open QuickTime → File → New Audio Recording → save as M4A

---

## 📦 Dependencies Overview

```
streamlit==1.35.0          # Web UI
requests==2.31.0           # HTTP client
python-dotenv==1.0.0       # Environment config
transformers==4.41.0       # DistilBERT emotion model
torch>=2.6.0               # PyTorch backend for transformers
SpeechRecognition==3.10.4  # Voice-to-text
chromadb==0.5.0            # Semantic vector memory
sentence-transformers==3.0.0  # Embeddings for ChromaDB
reportlab==4.2.0           # PDF report generation
```

---

## 🗂️ Features at a Glance

### 💬 Chat Tab
- Type or upload audio to interact
- Dynamic emotion-themed UI (colors, gradients update in real time)
- Context-aware empathetic LLM responses
- Chat history displayed per session

### 🌿 Suggestions Tab
- Personalized wellness tips based on detected emotion
- Quick relief techniques (Box Breathing, 5-4-3-2-1 Grounding, etc.)
- Emotional trend view from recent sessions

### 📈 Analytics Tab
- Mood frequency bars
- Dominant emotion detection
- Session timeline with timestamps
- ChromaDB vs JSON memory backend info

### 📄 Report Tab
- One-click PDF export
- Includes: stats, emotion breakdown, timeline, personalized insights
- All data stays local — fully private

---

## 🔒 Privacy

EmpathyOS is **fully on-device**. No emotional data is sent to any external server unless your LLM provider requires an API call. ChromaDB stores memory locally in the `chroma_memory/` folder.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `torch` install is slow | Use `pip install torch --index-url https://download.pytorch.org/whl/cpu` for CPU-only |
| ChromaDB errors on first run | Delete the `chroma_memory/` folder and restart |
| PDF export fails | Run `pip install reportlab` |
| Transcription not working | Make sure your audio file is WAV or MP3 format |
| PyAudio install fails on Windows | Use `pipwin install pyaudio` instead |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute.
