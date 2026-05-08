"""
EmpathyOS — v2
Full build with:
  1. DistilBERT emotion detection (+ keyword fallback)
  2. Voice input (mic recording + audio file upload)
  3. ChromaDB semantic memory (+ JSON fallback)
  4. PDF mood report export
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import datetime
from emotion_engine import detect_emotion, get_emotion_color, get_emotion_suggestions, get_detection_method
from memory_engine  import MemoryEngine
from llm_engine     import get_empathetic_response
from pdf_engine     import generate_mood_report
from voice_engine   import transcribe_audio, analyse_vocal_tone, check_dependencies

# ── Page Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmpathyOS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session State ────────────────────────────────────────────────────────────────
defaults = {
    "memory":          None,
    "chat_history":    [],
    "current_emotion": "neutral",
    "streak":          0,
    "mood_log":        [],
    "input_mode":      "text",
    "voice_transcript":"",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.memory is None:
    st.session_state.memory = MemoryEngine()

# ── Theme map ────────────────────────────────────────────────────────────────────
emotion    = st.session_state.current_emotion
COLOR_MAP  = {
    "happy":   {"bg": "#FFFBEB", "accent": "#F5A623", "emoji": "😊", "grad": "#FFF3CD"},
    "sad":     {"bg": "#EEF2FF", "accent": "#6B7FD7", "emoji": "😢", "grad": "#E0E7FF"},
    "angry":   {"bg": "#FFF0F0", "accent": "#E05555", "emoji": "😠", "grad": "#FFE4E4"},
    "anxious": {"bg": "#F0FFF4", "accent": "#48BB78", "emoji": "😰", "grad": "#DCFCE7"},
    "stressed":{"bg": "#FFF5F0", "accent": "#ED8936", "emoji": "😤", "grad": "#FFEDD5"},
    "excited": {"bg": "#FEFCE8", "accent": "#D69E2E", "emoji": "🤩", "grad": "#FEF9C3"},
    "neutral": {"bg": "#F8F9FA", "accent": "#667EEA", "emoji": "😐", "grad": "#EEF2FF"},
}
theme = COLOR_MAP.get(emotion, COLOR_MAP["neutral"])

st.markdown(f"""
<style>
  .stApp {{ background-color: {theme['bg']}; }}
  .main-header {{
    background: linear-gradient(135deg, {theme['accent']}, {theme['grad']});
    padding: 18px 24px; border-radius: 14px;
    margin-bottom: 12px;
  }}
  .main-title  {{ font-size:2rem; font-weight:800; color:#1A1A2E; margin:0; }}
  .main-sub    {{ font-size:0.95rem; color:#44445A; margin:0; }}
  .emotion-pill {{
    display:inline-block;
    background:{theme['accent']}22; color:{theme['accent']};
    padding:3px 14px; border-radius:20px;
    font-weight:700; font-size:0.88rem;
    border:1.5px solid {theme['accent']};
  }}
  .chat-user {{
    background:{theme['accent']}18;
    border-left:4px solid {theme['accent']};
    padding:10px 14px; border-radius:0 10px 10px 0;
    margin:6px 0; font-size:0.95rem;
  }}
  .chat-ai {{
    background:white; border-left:4px solid #CBD5E1;
    padding:10px 14px; border-radius:0 10px 10px 0;
    margin:6px 0; font-size:0.95rem;
  }}
  .card {{
    background:white; border:1.5px solid {theme['accent']}33;
    border-radius:12px; padding:14px; margin:6px 0;
  }}
  .metric-box {{
    background:white; border-radius:12px;
    padding:12px; text-align:center;
    border:1.5px solid {theme['accent']}33;
  }}
  .stButton>button {{
    background:{theme['accent']} !important;
    color:white !important; border:none !important;
    border-radius:10px !important; font-weight:600 !important;
    padding:8px 18px !important;
  }}
  .stTextInput>div>div>input {{
    border:2px solid {theme['accent']}66 !important;
    border-radius:10px !important;
  }}
  .stTabs [data-baseweb="tab-list"] {{ gap:8px; }}
  .stTabs [data-baseweb="tab"] {{
    border-radius:8px 8px 0 0 !important;
    padding:8px 18px !important;
  }}
  div[data-testid="metric-container"] {{
    background:white; border-radius:12px;
    border:1.5px solid {theme['accent']}33;
    padding:10px;
  }}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:10px 0 6px;">
      <span style="font-size:2.5rem;">🧠</span>
      <div style="font-size:1.4rem; font-weight:800; color:#1A1A2E;">EmpathyOS</div>
      <div style="font-size:0.8rem; color:#667788;">Emotional Intelligence, On-Device</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='text-align:center; margin:6px 0;'><span class='emotion-pill'>{theme['emoji']} {emotion.capitalize()}</span></div>", unsafe_allow_html=True)
    st.markdown("---")

    # Stats
    mem_info = st.session_state.memory.get_storage_info()
    c1, c2 = st.columns(2)
    c1.metric("🔥 Streak",    f"{st.session_state.streak}d")
    c2.metric("💬 Messages",  len(st.session_state.chat_history) // 2)

    c3, c4 = st.columns(2)
    c3.metric("📝 Memories",  mem_info["total_entries"])
    c4.metric("🧩 Backend",   "Chroma" if "ChromaDB" in mem_info["backend"] else "JSON")

    st.markdown("---")

    # Detection method badge
    method = get_detection_method()
    color  = "#48BB78" if "DistilBERT" in method else "#F59E0B"
    st.markdown(f"<div class='card' style='font-size:0.85rem;'>🔬 <b>Detection:</b> <span style='color:{color};'>{method}</span></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Mood history
    st.markdown("**🗓️ Recent Moods**")
    if st.session_state.mood_log:
        for entry in st.session_state.mood_log[-6:][::-1]:
            ec = COLOR_MAP.get(entry["emotion"], COLOR_MAP["neutral"])
            st.markdown(
                f"<div style='font-size:0.8rem; color:#555; padding:2px 0;'>"
                f"{ec['emoji']} <b>{entry['emotion'].capitalize()}</b> · {entry['time']}</div>",
                unsafe_allow_html=True
            )
    else:
        st.caption("Start chatting to track moods!")

    st.markdown("---")

    # Input mode toggle
    st.markdown("**🎤 Input Mode**")
    st.session_state.input_mode = st.radio(
        "Input mode", ["text", "voice"],
        format_func=lambda x: "⌨️ Text" if x == "text" else "🎙️ Voice",
        label_visibility="collapsed"
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Main Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <div class="main-title">🧠 EmpathyOS {theme['emoji']}</div>
  <div class="main-sub">Your on-device empathetic AI companion · Feeling <b>{emotion}</b></div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat", "🌿 Suggestions", "📈 Analytics", "📄 Report"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    # Display chat history
    for msg in st.session_state.chat_history:
        role_class = "chat-user" if msg["role"] == "user" else "chat-ai"
        icon       = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(
            f"<div class='{role_class}'>{icon} {msg['content']}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TEXT INPUT MODE ───────────────────────────────────────────────────────
    if st.session_state.input_mode == "text":
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Message",
                placeholder="How are you feeling right now? I'm here to listen 💙",
                label_visibility="collapsed"
            )
            col1, col2 = st.columns([5, 1])
            with col2:
                submitted = st.form_submit_button("Send 💬")

        if submitted and user_input.strip():
            _process_input(user_input)

    # ── VOICE INPUT MODE ──────────────────────────────────────────────────────
    else:
        deps = check_dependencies()
        st.markdown("### 🎙️ Voice Input")

        st.info("📁 **Upload an audio file** (WAV or MP3) — recorded from your phone/computer mic")

        uploaded_audio = st.file_uploader(
            "Upload audio", type=["wav", "mp3", "ogg", "m4a"],
            label_visibility="collapsed"
        )

        if uploaded_audio is not None:
            audio_bytes = uploaded_audio.read()
            st.audio(audio_bytes)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔊 Transcribe & Send"):
                    with st.spinner("Transcribing..."):
                        result = transcribe_audio(audio_bytes)
                    if result["error"]:
                        st.error(f"❌ {result['error']}")
                    else:
                        transcript = result["text"]
                        st.success(f"📝 Transcript: *{transcript}*")

                        # Vocal tone analysis
                        try:
                            tone_info = analyse_vocal_tone(audio_bytes)
                            if not tone_info["error"]:
                                st.caption(f"🎵 Vocal tone: **{tone_info['tone_hint']}** · Duration: {tone_info['duration_sec']}s")
                        except Exception:
                            pass

                        if transcript:
                            _process_input(transcript)
            with col2:
                st.markdown("""
                <div class='card' style='font-size:0.82rem;'>
                <b>📱 How to record:</b><br>
                • Phone: Voice Memo app → share as WAV<br>
                • Windows: Voice Recorder app<br>
                • Mac: QuickTime → New Audio Recording
                </div>
                """, unsafe_allow_html=True)

        # Live mic fallback info
        if not deps["pyaudio"]:
            st.markdown("""
            <div class='card' style='border-color:#F59E0B; font-size:0.85rem;'>
            💡 <b>Want live mic recording?</b><br>
            Run: <code>pip install pyaudio</code> then restart the app.<br>
            On Windows: <code>pip install pipwin && pipwin install pyaudio</code>
            </div>
            """, unsafe_allow_html=True)


def _process_input(user_text: str):
    """Shared logic for processing text or voice input"""
    detected  = detect_emotion(user_text)
    st.session_state.current_emotion = detected

    # Log mood
    st.session_state.mood_log.append({
        "emotion": detected,
        "time":    datetime.datetime.now().strftime("%I:%M %p"),
        "date":    datetime.datetime.now().strftime("%b %d"),
        "text":    user_text
    })
    st.session_state.streak += 1

    # Save to memory + get context
    st.session_state.memory.add(user_text, detected)
    context = st.session_state.memory.get_context()

    # Semantic memory search
    similar = st.session_state.memory.search_similar(user_text, n_results=3)

    # Get AI response
    response = get_empathetic_response(
        user_text, detected, context,
        st.session_state.chat_history, similar
    )

    st.session_state.chat_history.append({"role": "user",      "content": user_text})
    st.session_state.chat_history.append({"role": "assistant",  "content": response})
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUGGESTIONS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"### 🌿 Personalised for: {theme['emoji']} {emotion.capitalize()}")

    suggestions = get_emotion_suggestions(emotion)
    icons       = ["🎯", "🌱", "💡", "🎵", "🏃", "📖", "🧘", "💧"]

    for i, s in enumerate(suggestions):
        st.markdown(f"<div class='card'>{icons[i % len(icons)]} {s}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🆘 Quick Relief Technique")

    technique_map = {
        "anxious":  ("Box Breathing", "Inhale 4s → Hold 4s → Exhale 4s → Hold 4s. Repeat 4 times. This activates the parasympathetic nervous system."),
        "stressed": ("5-4-3-2-1 Grounding", "Name 5 things you see · 4 you can touch · 3 you hear · 2 you smell · 1 you taste."),
        "sad":      ("Gratitude Pause", "Write 3 small things that went okay today — even tiny ones count. This rewires the negativity bias."),
        "angry":    ("Cooldown Walk", "Step away for 5 minutes. Physical movement resets the stress hormone cycle."),
        "happy":    ("Savour It", "Write down what made you happy today — it reinforces positive neural pathways."),
        "excited":  ("Channel It", "Direct that energy into something creative before it fades. Sketch, write, or build something."),
        "neutral":  ("Mindful Check-in", "Take 3 deep breaths and ask: what do I truly need right now?"),
    }

    title, desc = technique_map.get(emotion, technique_map["neutral"])
    st.info(f"**{title}**\n\n{desc}")

    # Trend insight
    trend = st.session_state.memory.get_emotion_trend(5)
    if len(trend) >= 3:
        st.markdown("---")
        st.markdown("### 📊 Your Emotional Trend")
        trend_str = " → ".join([f"{COLOR_MAP.get(e, COLOR_MAP['neutral'])['emoji']} {e}" for e in trend])
        st.markdown(f"<div class='card' style='font-size:0.9rem;'>{trend_str}</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Your Mood Journey")

    if len(st.session_state.mood_log) < 2:
        st.info("💡 Need at least 2 check-ins to show analytics. Start chatting!")
    else:
        log     = st.session_state.mood_log
        counts  = {}
        for e in log:
            counts[e["emotion"]] = counts.get(e["emotion"], 0) + 1

        total = len(log)

        # Top-level stats
        dominant = max(counts, key=counts.get)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Check-ins",  total)
        c2.metric("Dominant Mood",    f"{COLOR_MAP[dominant]['emoji']} {dominant.capitalize()}")
        c3.metric("Unique Emotions",  len(counts))
        c4.metric("🔥 Streak",        f"{st.session_state.streak} days")

        st.markdown("---")

        # Emotion frequency bars
        st.markdown("#### Emotion Frequency")
        for em, count in sorted(counts.items(), key=lambda x: -x[1]):
            ec  = COLOR_MAP.get(em, COLOR_MAP["neutral"])
            pct = int((count / total) * 100)
            st.markdown(f"""
            <div style='display:flex; align-items:center; margin:5px 0; gap:10px;'>
              <span style='width:100px; font-size:0.9rem;'>{ec['emoji']} {em.capitalize()}</span>
              <div style='flex:1; background:#eee; border-radius:6px; height:18px;'>
                <div style='width:{pct}%; background:{ec['accent']}; height:18px; border-radius:6px;'></div>
              </div>
              <span style='width:50px; font-size:0.85rem; color:#555;'>{count}x · {pct}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Timeline
        st.markdown("#### Recent Timeline")
        for entry in log[-10:][::-1]:
            ec = COLOR_MAP.get(entry["emotion"], COLOR_MAP["neutral"])
            text_preview = entry["text"][:70] + ("..." if len(entry["text"]) > 70 else "")
            st.markdown(f"""
            <div style='padding:7px 12px; border-left:3px solid {ec['accent']};
                        margin:4px 0; background:white; border-radius:0 8px 8px 0;'>
              <b style='color:{ec['accent']};'>{ec['emoji']} {entry['emotion'].capitalize()}</b>
              <span style='color:#999; font-size:0.8rem; margin-left:8px;'>{entry['time']} · {entry['date']}</span>
              <div style='color:#555; font-size:0.88rem; margin-top:2px;'><i>{text_preview}</i></div>
            </div>
            """, unsafe_allow_html=True)

        # Semantic memory panel
        st.markdown("---")
        st.markdown("#### 🧠 Memory Backend")
        mem_info = st.session_state.memory.get_storage_info()
        backend_color = "#48BB78" if "ChromaDB" in mem_info["backend"] else "#F59E0B"
        st.markdown(f"""
        <div class='card'>
          <b>Storage:</b> <span style='color:{backend_color};'>{mem_info['backend']}</span><br>
          <b>Total memories:</b> {mem_info['total_entries']}<br>
          <b>In session:</b> {mem_info['short_term']}
        </div>
        """, unsafe_allow_html=True)

        # Insight
        st.markdown("---")
        insight_map = {
            "happy":   "You've been in a great headspace! Keep doing what's working.",
            "sad":     "You've been carrying some heaviness. Reaching out is the first step.",
            "anxious": "Anxiety has been present. Grounding techniques and limiting news/social media help.",
            "stressed":"High stress detected. Break tasks into smaller steps and protect your breaks.",
            "angry":   "Frustration has been high. Physical activity releases stored emotional tension.",
            "excited": "High energy waves — channel them into something meaningful!",
            "neutral": "Balanced state — great time to plan, reflect, and set intentions.",
        }
        st.success(f"**Your Insight:** {COLOR_MAP[dominant]['emoji']} {dominant.capitalize()} — {insight_map.get(dominant, '')}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — PDF REPORT
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📄 Export Your Mood Report")
    st.markdown("Generate a beautiful PDF summary of your emotional journey.")

    if len(st.session_state.mood_log) == 0:
        st.info("💡 No mood data yet — start chatting to generate a report!")
    else:
        # Preview stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Check-ins",  len(st.session_state.mood_log))
        col2.metric("Streak",     f"{st.session_state.streak} days 🔥")
        col3.metric("Sessions",   len(st.session_state.chat_history) // 2)

        st.markdown("---")

        if st.button("📥 Generate & Download PDF Report"):
            with st.spinner("Generating your report..."):
                try:
                    pdf_bytes = generate_mood_report(
                        st.session_state.mood_log,
                        st.session_state.streak,
                        len(st.session_state.chat_history) // 2
                    )
                    filename = f"EmpathyOS_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf"
                    )
                    st.success("✅ Report ready! Click Download PDF above.")
                except Exception as e:
                    st.error(f"Error generating report: {e}\n\nRun: pip install reportlab")

        st.markdown("---")
        st.markdown("""
        <div class='card' style='font-size:0.85rem;'>
        <b>📦 Report includes:</b><br>
        ✅ Summary stats (check-ins, streak, dominant mood)<br>
        ✅ Emotion breakdown with percentage bars<br>
        ✅ Full mood timeline (last 15 entries)<br>
        ✅ Personalised insight based on your patterns<br>
        ✅ Privacy note: all data stays on your device
        </div>
        """, unsafe_allow_html=True)
